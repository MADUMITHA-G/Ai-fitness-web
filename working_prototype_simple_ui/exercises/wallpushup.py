# processors/wallpushup.py
import time
from collections import deque
import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate angle (degrees) between points a-b-c (a,b,c are (x,y) tuples or lists).
    """
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

class WallPushupCounter:
    """
    Wall pushup processor class: call .process(frame_bgr) -> (processed_frame_bgr, count)
    Keeps internal state (count, stage) so repeated calls continue counting.
    """
    def __init__(self, down_thresh=110.0, up_thresh=160.0, min_dt=0.35):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.min_dt = min_dt

        self.count = 0
        self.stage = "up"  # or "down"
        self._last_time = 0.0

        # per-instance MediaPipe Pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def reset(self):
        self.count = 0
        self.stage = "up"
        self._last_time = 0.0

    def _process_landmarks(self, lm, frame):
        """
        lm: results.pose_landmarks.landmark
        frame: BGR frame for pixel-size calculations
        returns: angle (float), stage (str), rep_done (bool), elbow_px (tuple) or None
        """
        h, w = frame.shape[:2]
        try:
            # get left & right arm coords (x,y)
            sr = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            er = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wr = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            sl = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wl = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        except Exception:
            return None, self.stage, False, None

        # compute angles (in normalized coords) then average
        angle_r = calculate_angle(sr, er, wr)
        angle_l = calculate_angle(sl, el, wl)
        angle = (angle_r + angle_l) / 2.0

        rep_done = False
        now = time.time()
        # threshold logic follows your original: down when angle < 110, rep when angle > 160 from down
        if angle < self.down_thresh:
            self.stage = "down"
        if angle > self.up_thresh and self.stage == "down":
            if (now - self._last_time) >= self.min_dt:
                self.count += 1
                rep_done = True
                self._last_time = now
            self.stage = "up"

        # pixel coordinate for placing angle text (use elbow average)
        elbow_x = int(((er[0] + el[0]) / 2.0) * w)
        elbow_y = int(((er[1] + el[1]) / 2.0) * h)
        elbow_px = (elbow_x, elbow_y)

        return angle, self.stage, rep_done, elbow_px

    def _draw_hud(self, frame, angle, stage, elbow_px, fps=0.0):
        """Draw count, stage and angle similar to your other processors."""
        try:
            cv2.rectangle(frame, (10, 10), (280, 130), (0, 0, 0), -1)
            cv2.putText(frame, f"Pushups: {self.count}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Stage: {stage}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            if angle is not None:
                cv2.putText(frame, f"Elbow: {int(angle)} deg", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            if fps:
                cv2.putText(frame, f"{fps:.1f} FPS", (10, frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            # draw elbow location
            if elbow_px:
                cv2.circle(frame, elbow_px, 6, (0,255,0), -1)
        except Exception:
            pass

    def process(self, frame_bgr):
        """
        Main entry. Accepts BGR frame (numpy array), returns (processed_bgr, count).
        """
        # mirror to keep same UX as other processors
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # update counter from landmarks
        angle, stage, rep_done, elbow_px = None, self.stage, False, None
        if results.pose_landmarks:
            try:
                angle, stage, rep_done, elbow_px = self._process_landmarks(results.pose_landmarks.landmark, frame)
            except Exception:
                pass

        # draw HUD
        self._draw_hud(frame, angle, stage, elbow_px, fps=0.0)

        return frame, self.count
