from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import os
import base64
import sqlite3
import cv2
import numpy as np
import time

# Import exercise processors
from exercises.curl import CurlCounter
from exercises.squat import SquatCounter, draw_hud
from exercises.wallpushup import WallPushupCounter
from exercises.high_stepping import HighStepCounter
from exercises.crunches import CrunchCounter

workouts_bp = Blueprint('workouts', __name__)

# Local DB connector (kept independent to avoid circular imports)
def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'app.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def decode_data_url_image(data_url: str):
    try:
        if not data_url or 'base64,' not in data_url:
            return None
        b64 = data_url.split('base64,', 1)[1]
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame_bgr
    except Exception:
        return None

# Processor instances
curl_counter = CurlCounter()
squat_counter = SquatCounter()
wallpush_counter = WallPushupCounter()
highstep_counter = HighStepCounter()
crunch_counter = CrunchCounter()


REGISTRY = {
    'curl': {
        'template': 'curl.html',
        'table': 'curls',
        'processor': curl_counter,
        'get_count': lambda: curl_counter.counter,
    },
    'squat': {
        'template': 'squat.html',
        'table': 'squats',
        'processor': squat_counter,
        'get_count': lambda: getattr(squat_counter, 'count', 0),
    },
    'wallpushup': {
        'template': 'wallpushup.html',
        'table': 'wallpushups',
        'processor': wallpush_counter,
        'get_count': lambda: wallpush_counter.count,
    },
    'highstep': {
        'template': 'workout_highstep.html',
        'table': 'highsteps',
        'processor': highstep_counter,
        'get_count': lambda: highstep_counter.counter,
    },
    'crunches': {
        'template': 'crunches.html',
        'table': 'crunches',
        'processor': crunch_counter,
        'get_count': lambda: crunch_counter.counter,
    },
}


@workouts_bp.route('/workout/<exercise>')
def workout(exercise: str):
    if 'name' not in session:
        return redirect(url_for('login'))
    meta = REGISTRY.get(exercise)
    if not meta:
        return redirect(url_for('exercise'))
    process_url = url_for('workouts.process_frame', exercise=exercise)
    return render_template(meta['template'], name=session.get('name'), processUrl=process_url)


@workouts_bp.route('/process/<exercise>', methods=['POST'])
def process_frame(exercise: str):
    meta = REGISTRY.get(exercise)
    if not meta:
        return jsonify({'error': 'unsupported_exercise'}), 400

    data = request.get_json() or {}
    frame_bgr = decode_data_url_image(data.get('image', ''))
    if frame_bgr is None:
        return jsonify({'image': data.get('image', ''), 'count': 0})

    # gate counting by session active flag
    is_active = False
    try:
        sess = session.get('exercise_session')
        is_active = bool(session.get('exercise_active') and sess and sess.get('exercise') == exercise)
    except Exception:
        is_active = False

    processor = meta['processor']
    try:
        if is_active:
            processed_frame, count = processor.process(frame_bgr)
        else:
            processed_frame = cv2.flip(frame_bgr, 1)
            count = meta['get_count']()
    except Exception:
        processed_frame = frame_bgr
        count = meta['get_count']()

    ok, enc = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data.get('image', ''), 'count': count})
    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{b64_out}', 'count': count})


def get_current_user_id():
    email = session.get('email')
    if not email:
        return None
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM register WHERE email = ?", (email,))
        row = cur.fetchone()
        cur.close(); conn.close()
        return dict(row)['user_id'] if row else None
    except Exception:
        try:
            cur.close(); conn.close()
        except Exception:
            pass
        return None


@workouts_bp.route('/start_exercise', methods=['POST'])
def start_exercise():
    if 'email' not in session:
        return jsonify({'error': 'not_logged_in'}), 401

    data = request.get_json() or {}
    exercise = data.get('exercise')
    meta = REGISTRY.get(exercise)
    if not meta:
        return jsonify({'error': 'unsupported_exercise'}), 400

    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'error': 'user_not_found'}), 400

    conn = get_db_connection(); cur = conn.cursor()
    initial_count = meta['get_count']()
    cur.execute(f"INSERT INTO {meta['table']} (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                (user_id, initial_count, 0))
    row_id = cur.lastrowid
    conn.commit(); cur.close(); conn.close()

    session['exercise_session'] = {
        'exercise': exercise,
        'row_id': row_id,
        'start_ts': int(time.time())
    }
    session['exercise_active'] = True

    return jsonify({'ok': True, 'row_id': row_id})


@workouts_bp.route('/stop_exercise', methods=['POST'])
def stop_exercise():
    if 'email' not in session:
        return jsonify({'error': 'not_logged_in'}), 401
    data = request.get_json() or {}
    exercise = data.get('exercise')
    meta = REGISTRY.get(exercise)
    if not meta:
        return jsonify({'error': 'unsupported_exercise'}), 400

    sess = session.pop('exercise_session', None)
    session.pop('exercise_active', None)
    session.pop('exercise_session_last_update', None)
    if not sess or sess.get('exercise') != exercise:
        return jsonify({'error': 'no_active_session_for_exercise'}), 400

    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'error': 'user_not_found'}), 400

    row_id = sess.get('row_id')
    start_ts = sess.get('start_ts')
    duration_seconds = int(time.time() - start_ts) if start_ts else 0
    timing_minutes = int(round(duration_seconds/60))

    # final reps
    final_count = meta['get_count']()

    conn = get_db_connection(); cur = conn.cursor()
    try:
        cur.execute(
            f"UPDATE {meta['table']} SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
            (final_count, timing_minutes, row_id, user_id)
        )
        conn.commit()
    finally:
        cur.close(); conn.close()

    return jsonify({'ok': True, 'duration_minutes': timing_minutes, 'reps': final_count})


