# processors/curl.py
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
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class CurlCounter:
    """
    CurlCounter processes single BGR frames and keeps internal state.
    Use .process(frame_bgr) -> (processed_frame_bgr, count)
    Use .reset() to reset the counter/state.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.counter = 0
        self.stage = None  # "up" or "down"
        self._last_time = 0.0

        # create per-instance MediaPipe Pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def reset(self):
        self.counter = 0
        self.stage = None
        self._last_time = 0.0

    def update_from_results(self, results, frame):
        """
        Uses results (MediaPipe) and original frame to update state.
        Returns (angle, stage, rep_done)
        """
        if results.pose_landmarks is None:
            return None, self.stage, False

        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # using LEFT side (like original). You can extend for right side too.
        try:
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        except Exception:
            return None, self.stage, False

        angle = calculate_angle(shoulder, elbow, wrist)

        rep_done = False
        # original thresholds: >160 down, <30 up
        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == "down":
            self.stage = "up"
            # optional debounce based on time (mirrors original behaviour)
            now = time.time()
            if now - self._last_time > 0.3:   # small debounce
                self.counter += 1
                rep_done = True
                self._last_time = now

        # compute pixel coords for elbow to show angle
        elbow_px = (int(elbow[0] * w), int(elbow[1] * h))

        return angle, self.stage, rep_done, elbow_px

    def draw_hud(self, frame, angle, stage, elbow_px):
        """Draw the same HUD as original script."""
        try:
            # status box
            cv2.rectangle(frame, (0,0), (230,90), (245,117,16), -1)
            cv2.putText(frame, 'REPS', (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.counter),
                        (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(frame, 'STAGE', (65,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(stage if stage is not None else ""),
                        (60,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # angle near elbow
            if elbow_px and angle is not None:
                cv2.putText(frame, str(int(angle)),
                            (elbow_px[0], elbow_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        except Exception:
            pass

    def process(self, frame_bgr):
        """
        Accepts a BGR frame (numpy array), returns (processed_bgr, count).
        This mimics the original script behavior but for single-frame processing.
        """
        # mirror to keep UX consistent
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # draw landmarks (style similar to original)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        angle, stage, rep_done, elbow_px = None, self.stage, False, None
        try:
            out = self.update_from_results(results, frame)
            if out is not None:
                angle, stage, rep_done, elbow_px = out
        except Exception:
            pass

        # draw HUD
        self.draw_hud(frame, angle, stage, elbow_px)

        return frame, self.counter
