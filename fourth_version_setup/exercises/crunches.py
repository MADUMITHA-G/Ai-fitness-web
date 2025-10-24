import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class CrunchCounter:
    """
    CrunchCounter processes BGR frames and keeps internal rep state. Use process(frame_bgr) and reset() as with other exercises.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.counter = 0
        self.stage = None 
        self._last_time = 0.0
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def reset(self):
        self.counter = 0
        self.stage = None
        self._last_time = 0.0

    def update_from_results(self, results, frame):
        if results.pose_landmarks is None:
            return None, self.stage, False
        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        try:
            # Use hip-shoulder-knee (abdominal crunch angle)
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        except Exception:
            return None, self.stage, False
        angle = calculate_angle(shoulder, hip, knee)
        rep_done = False
        # Rep = crunch up below threshold, then back to start pos (angle expands)
        if angle > 135:
            self.stage = "down"
        if angle < 85 and self.stage == "down":
            self.stage = "up"
            now = time.time()
            if now - self._last_time > 0.4:
                self.counter += 1
                rep_done = True
                self._last_time = now
        hip_px = (int(hip[0]*w), int(hip[1]*h))
        return angle, self.stage, rep_done, hip_px

    def draw_hud(self, frame, angle, stage, hip_px):
        try:
            cv2.rectangle(frame, (0, 0), (230, 90), (245, 117, 16), -1)
            cv2.putText(frame, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.counter), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(stage if stage is not None else ""), (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if hip_px and angle is not None:
                cv2.putText(frame, str(int(angle)), (hip_px[0], hip_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

    def process(self, frame_bgr):
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        angle, stage, rep_done, hip_px = None, self.stage, False, None
        try:
            out = self.update_from_results(results, frame)
            if out is not None:
                angle, stage, rep_done, hip_px = out
        except Exception:
            pass
        self.draw_hud(frame, angle, stage, hip_px)
        return frame, self.counter
