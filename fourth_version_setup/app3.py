import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify ,abort
import cv2
import mediapipe as mp
import numpy as np
import base64
import re
import os
from pathlib import Path

from exercises.squat import SquatCounter, draw_hud
from exercises.curl import CurlCounter
from exercises.wallpushup import WallPushupCounter
from exercises.placeholder import PlaceholderProcessor
from workouts import workouts_bp

placeholder_proc = PlaceholderProcessor()
curl_counter = CurlCounter()
wallpush_counter = WallPushupCounter()



def get_db_connection():
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'app.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as err:
        print(f"Error: Could not connect to SQLite. {err}")
        return None

def init_db():
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'app.db')
        sql_path = Path(os.path.join(os.path.dirname(__file__), 'db.sql'))
        if sql_path.exists():
            conn = sqlite3.connect(db_path)
            with open(sql_path, 'r', encoding='utf-8') as f:
                script = f.read()
            conn.executescript(script)
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"DB init error: {e}")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.register_blueprint(workouts_bp)

# ---------------------- DIET ROUTE ----------------------
@app.route("/diet/day<int:day>")
def diet_day(day):
    """
    Dynamically loads each day's diet plan.
    Example: /diet/day1 loads templates/diet_page/day1.html
    """
    if 1 <= day <= 30:
        return render_template(f"diet_page/day{day}.html")
    else:
        abort(404)
# --------------------------------------------------------


# --- MediaPipe Pose setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

data_url_pattern = re.compile(r'^data:image/.+;base64,(.*)$')

def decode_data_url_image(data_url):
    m = data_url_pattern.match(data_url or '')
    if not m:
        return None
    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame_bgr
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# ---------- Squat-specific globals ----------
# Use a separate Pose instance for squat processing (keeps logic clear).
pose_squat = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# persistent in-memory counter object for squat (keeps state across requests)
squat_counter = SquatCounter()

@app.route('/')
def wel():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

#href="{{ url_for('back') }}"

@app.route('/exercise')
def exercise():
    if 'name' in session:
        #return render_template('dashboard.html', name=session['name'])
        return render_template('exercise_pages/exercise_home.html', name=session['name'])
    return redirect(url_for('login'))


@app.route('/legrise')
def legrise():
    return render_template('legrise.html')

@app.route('/muscle_building')
def muscle_building():
    if 'name' in session:
        return render_template('exercise_pages/muscle_building.html')
    return redirect(url_for('login'))

@app.route('/fullbody')
def fullbody():
    if 'name' in session:
        return render_template('exercise_pages/fullbody.html')
    return redirect(url_for('login'))

@app.route('/abs')
def abs_workout():
    if 'name' in session:
        return render_template('exercise_pages/abs_workout.html')
    return redirect(url_for('login'))

@app.route('/fatloss')
def fatloss():
    if 'name' in session:
        return render_template('exercise_pages/fatloss.html')
    return redirect(url_for('login'))



@app.route('/faq')
def faq():
    if 'name' in session:
        return render_template('faq.html')
    return redirect(url_for('login'))



@app.route('/men-workout')
def men_workout():
    return render_template('exercise_pages/men-workouts.html')

@app.route('/women-workout')
def women_workout():
    return render_template('exercise_pages/women-workouts.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot_api', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Imports (assuming you handle load_dotenv and os.environ elsewhere)
    from google import genai
    from google.genai import types
    import os # Make sure this is imported if using os.environ.get

    # Load API Key
    # Ensure load_dotenv() is called at the very top of your application file
    API_KEY = os.environ.get('GEMINI_API_KEY')
    if not API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY not configured'}), 500

    client = genai.Client(api_key=API_KEY)
    model = "gemini-2.5-flash-lite" # Fast, cost-effective, text-only output

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        ),
    ]

    # **FIX:** Removed response_modalities=["IMAGE", "TEXT"]
    # Simplified system_instruction passing
    generate_content_config = types.GenerateContentConfig(
        system_instruction="""You are a fitness trainer named FitBot, your task is to help people by clarifying their doubts regarding fitness and diet. Provide clear, actionable advice and encouraging feedback."""
    )
    
    response_text = ""
    # The generator streams text chunks
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # We only expect and process text
        if hasattr(chunk, 'text') and chunk.text:
             response_text += chunk.text
    
    # **FIX:** Removed image_data_uris from the return
    return jsonify({'text': response_text.strip()})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login and session creation."""
    # Clear any existing session data before a new login attempt
    session.pop('email', None)

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn is None:
            error = "Could not connect to the database. Please check your connection details."
            return render_template('login.html', error=error)
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM register WHERE email = ? AND password = ?", (email, password))
            user = cursor.fetchone()
            
            
            if user:
                session['email'] = user['email']
                session['name'] = user['name']
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid credentials. Please try again."
                return render_template('login.html', error=error)
        except sqlite3.Error as err:
            error = f"Database error: {err}"
            return render_template('login.html', error=error)
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles new user registration."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        age = request.form['age']
        gender = request.form['gender']
        height = request.form['height']
        weight = request.form['weight']

        if password != confirm_password:
            error = "Passwords do not match."
            return render_template('register.html', error=error)
        
        conn = get_db_connection()
        if conn is None:
            error = "Could not connect to the database. Please check your connection details."
            return render_template('register.html', error=error)

        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO register (name, email, password, confirm_password, age, gender, height, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (name, email, password, confirm_password, age, gender, height, weight))
            conn.commit()
            
            return redirect(url_for('login'))
        except sqlite3.Error as err:
            error = f"Database error: {err}"
            return render_template('register.html', error=error)
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """Displays the user's dashboard after successful login."""
    if 'name' in session:
        return render_template('exercise_pages/exercise_home.html', name=session['name']) 
        #return render_template('dashboard.html', name=session['name']) 
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Handles user logout and session clearing."""
    session.pop('email', None)
    session.pop('name', None)
    return redirect(url_for('home'))

@app.route('/workout_highstep')
def workout_highstep():
    # Backward compat: redirect to unified workout route
    return redirect(url_for('workouts.workout', exercise='highstep'))

# ------------------ NEW: Squat routes ------------------

@app.route('/workout_squat')
def workout_squat():
    return redirect(url_for('workouts.workout', exercise='squat'))


@app.route('/process_squat', methods=['POST'])
def process_squat():
    return jsonify({'error': 'Use /process/squat via workouts blueprint'}), 410

# ------------------ NEW: curk routes ------------------

@app.route('/process_curl', methods=['POST'])
def process_curl():
    return jsonify({'error': 'Use /process/curl via workouts blueprint'}), 410

#---------------------------------wallpushup----------------------------------------

@app.route('/workout_wallpushup')
def workout_wallpushup():
    return redirect(url_for('workouts.workout', exercise='wallpushup'))

@app.route('/process_wallpushup', methods=['POST'])
def process_wallpushup():
    return jsonify({'error': 'Use /process/wallpushup via workouts blueprint'}), 410

#----------------------------placeholder------------------------

@app.route('/workout_placeholder')
def workout_placeholder():
    """Serve placeholder workout page (shows landmarks only)."""
    if 'name' in session:
        session['placeholder_count'] = 0
        return render_template('placeholder.html', name=session['name'])
    return redirect(url_for('login'))


@app.route('/process_placeholder', methods=['POST'])
def process_placeholder():
    """
    Receives JSON { "image": "data:image/jpeg;base64,..." } and returns
    { "image": "data:image/jpeg;base64,...", "count": <int> }
    Uses processors/placeholder.py -> PlaceholderProcessor.process()
    """
    global placeholder_proc

    data = request.get_json()
    data_url = data.get('image', '')
    frame_bgr = decode_data_url_image(data_url)

    if frame_bgr is None:
        return jsonify({'image': data_url, 'count': 0})

    try:
        processed_frame, count = placeholder_proc.process(frame_bgr)
    except Exception as e:
        print(f"Error processing placeholder frame: {e}")
        processed_frame = frame_bgr
        count = 0

    ok, enc = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url, 'count': count})

    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({
        'image': f'data:image/jpeg;base64,{b64_out}',
        'count': count
    })


# Optional reset endpoint (no-op for placeholder but included for parity)
@app.route('/reset_placeholder', methods=['POST'])
def reset_placeholder():
    global placeholder_proc
    try:
        placeholder_proc.reset()
        session['placeholder_count'] = 0
        return jsonify({'ok': True})
    except Exception as e:
        print(f"Error resetting placeholder processor: {e}")
        return jsonify({'ok': False}), 500

#-----------------------------profile login---------------------

@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, height, weight FROM register WHERE email = ?", (session['email'],))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return redirect(url_for('dashboard'))

    user = dict(row)
    return render_template('profile.html', 
                           name=user.get('name'), 
                           age=user.get('age'), 
                           gender=user.get('gender'), 
                           height=user.get('height'), 
                           weight=user.get('weight'))

# ---------- Helper: get current user id ----------
import time  # add near top if not already imported

def get_current_user_id():
    """Return user_id from register for current session email, or None."""
    email = session.get('email')
    if not email:
        return None
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM register WHERE email = ?", (email,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row)['user_id'] if row else None
    except Exception as e:
        print("get_current_user_id error:", e)
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass
        return None

#B

@app.route('/start_exercise', methods=['POST'])
def start_exercise():
    """
    Create (or resume) a DB row for the exercise and mark session active.
    Request JSON: { "exercise": "curl" }
    Response: { "ok": True, "row_id": <id> }
    """
    if 'email' not in session:
        return jsonify({'error': 'not_logged_in'}), 401

    data = request.get_json() or {}
    exercise = data.get('exercise')
    if not exercise:
        return jsonify({'error': 'no_exercise_provided'}), 400

    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'error': 'user_not_found'}), 400

    # If there is already an active session for this exercise in Flask session, return it
    sess = session.get('exercise_session')
    if sess and sess.get('exercise') == exercise:
        return jsonify({'ok': True, 'row_id': sess.get('row_id')})

    # Create new DB row per exercise
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'db_conn_failed'}), 500
    cur = conn.cursor()

    # Determine current count from processor (resume)
    initial_count = 0
    if exercise == 'curl':
        initial_count = curl_counter.counter
        cur.execute("INSERT INTO curls (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                    (user_id, initial_count, 0))
        row_id = cur.lastrowid
    elif exercise == 'squat':
        initial_count = squat_counter.count if hasattr(squat_counter, 'count') else 0
        cur.execute("INSERT INTO squats (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                    (user_id, initial_count, 0))
        row_id = cur.lastrowid
    elif exercise == 'highstep':
        initial_count = session.get('counter', 0)
        cur.execute("INSERT INTO highsteps (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                    (user_id, initial_count, 0))
        row_id = cur.lastrowid
    elif exercise == 'wallpushup':
        initial_count = wallpush_counter.count
        cur.execute("INSERT INTO wallpushups (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                    (user_id, initial_count, 0))
        row_id = cur.lastrowid
    elif exercise == 'crunches':
        from exercises.crunches import CrunchCounter
        from workouts import crunch_counter
        initial_count = crunch_counter.counter
        cur.execute("INSERT INTO crunches (user_id, count, timing_minutes, session_day, start_ts) VALUES (?, ?, ?, DATE('now'), CURRENT_TIMESTAMP)",
                    (user_id, initial_count, 0))
        row_id = cur.lastrowid
    else:
        # Fallback for other exercises: create a placeholder record in appropriate table or return error
        cur.close(); conn.close()
        return jsonify({'error': 'unsupported_exercise'}), 400

    conn.commit()
    cur.close()
    conn.close()

    # Save session info so we can update later
    session['exercise_session'] = {
        'exercise': exercise,
        'row_id': row_id,
        'start_ts': int(time.time())
    }
    session['exercise_active'] = True

    return jsonify({'ok': True, 'row_id': row_id})

#c
@app.route('/stop_exercise', methods=['POST'])
def stop_exercise():
    """
    Finalize the DB row for the exercise and return summary.
    Request JSON: { "exercise": "curl" }
    Response JSON: { ok: True, duration_minutes: n, reps: m }
    """
    if 'email' not in session:
        return jsonify({'error': 'not_logged_in'}), 401

    data = request.get_json() or {}
    exercise = data.get('exercise')
    if not exercise:
        return jsonify({'error': 'no_exercise_provided'}), 400

    sess = session.pop('exercise_session', None)
    # remove the active flag
    session.pop('exercise_active', None)
    session.pop('exercise_session_last_update', None)

    if not sess or sess.get('exercise') != exercise:
        return jsonify({'error': 'no_active_session_for_exercise'}), 400

    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'error': 'user_not_found'}), 400

    row_id = sess.get('row_id')
    start_ts = sess.get('start_ts')
    duration_seconds = int(time.time() - start_ts) if start_ts else None
    timing_minutes = int(round(duration_seconds/60)) if duration_seconds is not None else 0

    # get final reps from processor
    final_count = 0
    if exercise == 'curl':
        final_count = curl_counter.counter
    elif exercise == 'squat':
        final_count = squat_counter.count
    elif exercise == 'highstep':
        final_count = session.get('counter', 0)
    elif exercise == 'wallpushup':
        final_count = wallpush_counter.count
    elif exercise == 'crunches':
        from exercises.crunches import CrunchCounter
        from workouts import crunch_counter
        final_count = crunch_counter.counter
    else:
        final_count = 0

    # update DB row
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'db_conn_failed'}), 500
    cur = conn.cursor()
    try:
        if exercise == 'curl':
            cur.execute(
                "UPDATE curls SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
                (final_count, timing_minutes, row_id, user_id)
            )
        elif exercise == 'squat':
            cur.execute(
                "UPDATE squats SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
                (final_count, timing_minutes, row_id, user_id)
            )
        elif exercise == 'highstep':
            cur.execute(
                "UPDATE highsteps SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
                (final_count, timing_minutes, row_id, user_id)
            )
        elif exercise == 'wallpushup':
            cur.execute(
                "UPDATE wallpushups SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
                (final_count, timing_minutes, row_id, user_id)
            )
        elif exercise == 'crunches':
            cur.execute(
                "UPDATE crunches SET count=?, timing_minutes=?, end_ts=CURRENT_TIMESTAMP, session_day=DATE('now') WHERE session_id=? AND user_id=?",
                (final_count, timing_minutes, row_id, user_id)
            )
        # add other exercises similarly
        conn.commit()
    except Exception as e:
        print("stop_exercise db update error:", e)
    finally:
        cur.close()
        conn.close()

    return jsonify({'ok': True, 'duration_minutes': timing_minutes, 'reps': final_count})

#d
@app.route('/workout_curl')
def workout_curl():
    """Serve curl workout page (create templates/curl.html similar to squat.html)."""
    if 'name' not in session:
        return redirect(url_for('login'))

    resume_count = 0
    sess = session.get('exercise_session')
    if sess and sess.get('exercise') == 'curl':
        # fetch DB row count to ensure sync
        row_id = sess.get('row_id')
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT count FROM curls WHERE session_id = ? AND user_id = (SELECT user_id FROM register WHERE email=?)", (row_id, session.get('email')))
                row = cur.fetchone()
                if row:
                    resume_count = dict(row).get('count', 0)
                    # set processor counter so UI resumes
                    try:
                        curl_counter.counter = int(resume_count)
                    except Exception:
                        curl_counter.counter = resume_count
                cur.close()
                conn.close()
            except Exception as e:
                print("workout_curl resume DB error:", e)
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass

    # optional initialize session value
    session['curl_count'] = curl_counter.counter
    return render_template('curl.html', name=session.get('name'), resume_count=curl_counter.counter)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, threaded=True)
