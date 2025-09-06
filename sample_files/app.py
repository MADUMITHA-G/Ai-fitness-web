from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import re
import mediapipe as mp

app = Flask(__name__)

# --- MediaPipe Hands setup (global so we don't reinitialize per frame) ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Tune these params as needed:
hands = mp_hands.Hands(
    static_image_mode=False,      # we get a stream; keep tracking state
    max_num_hands=2,
    model_complexity=1,           # 0 = faster, 1 = default, 2 = most accurate
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Regex to strip the data URL prefix
data_url_pattern = re.compile(r'^data:image/.+;base64,(.*)$')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    data_url = data.get('image', '')

    m = data_url_pattern.match(data_url)
    if not m:
        return jsonify({'image': data_url})  # just echo back if bad payload

    # Decode incoming JPEG (base64)
    img_bytes = base64.b64decode(m.group(1))
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({'image': data_url})

    # ---- MediaPipe Hand Tracking ----
    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr,                  # draw on BGR (display image)
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )

    # Encode processed frame back to JPEG
    ok, enc = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({'image': data_url})
    b64_out = base64.b64encode(enc.tobytes()).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{b64_out}'})

if __name__ == '__main__':
    # threaded=True helps when frames arrive quickly
    app.run(debug=True, threaded=True)
