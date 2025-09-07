import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import re

# Import SquatCounter and draw_hud from your provided squat.py
# Place squat.py in same folder as this app.py
from exercises.squat import SquatCounter, draw_hud
from exercises.curl import CurlCounter
from exercises.wallpushup import WallPushupCounter
# app.py (add near top with other imports)
from exercises.placeholder import PlaceholderProcessor


# create a global placeholder processor instance
placeholder_proc = PlaceholderProcessor()


# create a global instance (simple approach)
curl_counter = CurlCounter()
wallpush_counter = WallPushupCounter()

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="lotusGirl@05", # Replace with your MySQL password
            database="proto"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: Could not connect to MySQL. {err}")
        return None

app = Flask(__name__)
app.secret_key = 'victoryIsMine' # Replace with a random, secret key

# --- MediaPipe Pose setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Regex to strip the data URL prefix
data_url_pattern = re.compile(r'^data:image/.+;base64,(.*)$')

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
def home():
    """Redirects to the dashboard if a user is logged in, otherwise to login."""
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

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
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM register WHERE email = %s AND password = %s", (email, password))
            user = cursor.fetchone()
            
            if user:
                session['email'] = user['email']
                session['name'] = user['name']
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid credentials. Please try again."
                return render_template('login.html', error=error)
        except mysql.connector.Error as err:
            error = f"Database error: {err}"
            return render_template('login.html', error=error)
        finally:
            if conn and conn.is_connected():
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
            cursor.execute("INSERT INTO register (name, email, password, confirm_password, age, gender, height, weight) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                           (name, email, password, confirm_password, age, gender, height, weight))
            conn.commit()
            
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            error = f"Database error: {err}"
            return render_template('register.html', error=error)
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """Displays the user's dashboard after successful login."""
    if 'name' in session:
        return render_template('dashboard.html', name=session['name'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Handles user logout and session clearing."""
    session.pop('email', None)
    session.pop('name', None)
    return redirect(url_for('login'))

@app.route('/workout_highstep')
def workout_highstep():
    """Serves the high-step workout page and initializes session variables."""
    if 'name' in session:
        # Initialize session variables for the workout counter
        session['counter'] = 0
        session['left_stage'] = 'down'
        session['right_stage'] = 'down'
        return render_template('workout_highstep.html', name=session['name'])
    return redirect(url_for('login'))


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Processes a single frame from the camera feed using MediaPipe."""
    data = request.get_json()
    data_url = data.get('image', '')

    m = re.match(r'^data:image/.+;base64,(.*)$', data_url)
    if not m:
        return jsonify({'image': data_url, 'count': 0}) 

    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({'error': 'Failed to decode image'}), 400

    if frame_bgr is None:
        return jsonify({'image': data_url, 'count': 0})

    # Flip the frame horizontally for intuitive view
    frame_bgr = cv2.flip(frame_bgr, 1)

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    current_count = session.get('counter', 0)
    left_stage = session.get('left_stage', 'down')
    right_stage = session.get('right_stage', 'down')

    if results.pose_landmarks:
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            frame_bgr, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get y-coordinates for hip and knee joints
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            
            # Logic to count high steps
            if left_knee_y < left_hip_y - 0.1:
                if left_stage == 'down':
                    current_count += 1
                left_stage = 'up'
            else:
                left_stage = 'down'

            if right_knee_y < right_hip_y - 0.1:
                if right_stage == 'down':
                    current_count += 1
                right_stage = 'up'
            else:
                right_stage = 'down'
            
            # Update session variables
            session['counter'] = current_count
            session['left_stage'] = left_stage
            session['right_stage'] = right_stage
            
        except Exception as e:
            # Pass if landmarks are not detected
            print(f"Error in pose detection logic: {e}")
            pass

    # Encode processed frame back to JPEG
    ok, enc = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url, 'count': current_count})
        
    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({
        'image': f'data:image/jpeg;base64,{b64_out}',
        'count': current_count,
        'left_stage': left_stage,
        'right_stage': right_stage
    })

# ------------------ NEW: Squat routes ------------------

@app.route('/workout_squat')
def workout_squat():
    """Serves the squat workout page and initializes session variables."""
    if 'name' in session:
        # initialize session tracking for squat (optional)
        session['squat_count'] = 0
        return render_template('squat.html', name=session['name'])
    return redirect(url_for('login'))


@app.route('/process_squat', methods=['POST'])
def process_squat():
    """
    Receives base64 frame from squat.html, runs the SquatCounter logic
    (from squat.py), draws HUD/landmarks, and returns processed frame + count.
    """
    global squat_counter, pose_squat

    data = request.get_json()
    data_url = data.get('image', '')

    m = re.match(r'^data:image/.+;base64,(.*)$', data_url)
    if not m:
        return jsonify({'image': data_url, 'count': 0})

    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding squat image: {e}")
        return jsonify({'error': 'Failed to decode image'}), 400

    if frame_bgr is None:
        return jsonify({'image': data_url, 'count': 0})

    # Mirror for intuitive user view
    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Process with a dedicated MediaPipe pose for squat
    results = pose_squat.process(frame_rgb)

    # Draw landmarks if present
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
        )

    # Update squat counter using the class you provided
    try:
        angle, phase, rep_done = squat_counter.update(results)
        if rep_done:
            # update session copy (optional)
            session['squat_count'] = squat_counter.count
    except Exception as e:
        print(f"Error in squat counter update: {e}")
        # continue, don't crash

    # Draw HUD using your function (from squat.py)
    try:
        # Use fps=0 here (optional: you can compute time delta if you want)
        draw_hud(frame_bgr, squat_counter.count, phase if 'phase' in locals() else 'up', angle if 'angle' in locals() else 0, fps=0.0)
    except Exception as e:
        print(f"Error drawing HUD: {e}")

    # Encode and return processed image + count
    ok, enc = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url, 'count': squat_counter.count})

    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({
        'image': f'data:image/jpeg;base64,{b64_out}',
        'count': squat_counter.count
    })

# ------------------ NEW: curk routes ------------------

@app.route('/workout_curl')
def workout_curl():
    """Serve curl workout page (create templates/curl.html similar to squat.html)."""
    if 'name' in session:
        # optional initialize session value
        session['curl_count'] = 0
        return render_template('curl.html', name=session.get('name'))
    return redirect(url_for('login'))

@app.route('/process_curl', methods=['POST'])
def process_curl():
    """
    Receives JSON { "image": "data:image/jpeg;base64,..."} and returns
    {'image': 'data:image/jpeg;base64,...', 'count': <int>}
    """
    global curl_counter

    data = request.get_json()
    data_url = data.get('image', '')

    m = re.match(r'^data:image/.+;base64,(.*)$', data_url)
    if not m:
        return jsonify({'image': data_url, 'count': 0})

    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding curl image: {e}")
        return jsonify({'error': 'Failed to decode image'}), 400

    if frame_bgr is None:
        return jsonify({'image': data_url, 'count': 0})

    try:
        processed_frame, count = curl_counter.process(frame_bgr)
    except Exception as e:
        print(f"Error processing curl frame: {e}")
        processed_frame = frame_bgr
        count = curl_counter.counter

    ok, enc = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url, 'count': count})

    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    # backend returns full data:image/jpeg;base64,... like your other endpoints
    return jsonify({'image': f'data:image/jpeg;base64,{b64_out}', 'count': count})

#---------------------------------wallpushup----------------------------------------

@app.route('/workout_wallpushup')
def workout_wallpushup():
    if 'name' in session:
        session['wallpush_count'] = 0
        return render_template('wallpushup.html', name=session['name'])
    return redirect(url_for('login'))

@app.route('/process_wallpushup', methods=['POST'])
def process_wallpushup():
    global wallpush_counter

    data = request.get_json()
    data_url = data.get('image', '')

    m = re.match(r'^data:image/.+;base64,(.*)$', data_url)
    if not m:
        return jsonify({'image': data_url, 'count': 0})

    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding wallpush image: {e}")
        return jsonify({'error': 'Failed to decode image'}), 400

    if frame_bgr is None:
        return jsonify({'image': data_url, 'count': 0})

    try:
        processed_frame, count = wallpush_counter.process(frame_bgr)
        # keep optional session copy
        session['wallpush_count'] = count
    except Exception as e:
        print(f"Error processing wallpush frame: {e}")
        processed_frame = frame_bgr
        count = wallpush_counter.count

    ok, enc = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url, 'count': count})

    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{b64_out}', 'count': count})

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

    m = re.match(r'^data:image/.+;base64,(.*)$', data_url)
    if not m:
        return jsonify({'image': data_url, 'count': 0})

    try:
        img_bytes = base64.b64decode(m.group(1))
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding placeholder image: {e}")
        return jsonify({'error': 'Failed to decode image'}), 400

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
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, age, gender, height, weight FROM register WHERE email = %s", (session['email'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template('profile.html', 
                           name=user['name'], 
                           age=user['age'], 
                           gender=user['gender'], 
                           height=user['height'], 
                           weight=user['weight'])


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
