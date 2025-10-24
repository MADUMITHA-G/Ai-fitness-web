# processors/high_stepping.py
import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class HighStepCounter:
    """
    HighStepCounter: per-frame processing for high-step exercise.
    Use .process(frame_bgr) -> (processed_bgr, count)
    Use .reset() to reset internal state (counter/stages/debounce).
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, up_threshold=0.1, min_dt=0.3):
        # counter and state
        self.counter = 0
        self.left_stage = 'down'
        self.right_stage = 'down'
        self._last_time = 0.0
        self.min_dt = min_dt
        self.up_threshold = up_threshold  # how much knee must be above hip (in normalized coords)

        # per-instance MediaPipe Pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def reset(self):
        """Reset counter and state."""
        self.counter = 0
        self.left_stage = 'down'
        self.right_stage = 'down'
        self._last_time = 0.0

    def update_from_results(self, results):
        """
        Update counter/state from MediaPipe `results`.
        Returns: (rep_done: bool, left_stage, right_stage, landmarks) 
        landmarks is results.pose_landmarks.landmark or None.
        """
        if results is None or results.pose_landmarks is None:
            return False, self.left_stage, self.right_stage, None

        lm = results.pose_landmarks.landmark

        try:
            left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee_y = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y

            right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        except Exception:
            return False, self.left_stage, self.right_stage, lm

        rep_done = False
        now = time.time()

        # Left leg logic
        if left_knee_y < left_hip_y - self.up_threshold:
            if self.left_stage == 'down':
                # debounce by time to avoid double counting rapidly
                if now - self._last_time >= self.min_dt:
                    self.counter += 1
                    rep_done = True
                    self._last_time = now
            self.left_stage = 'up'
        else:
            self.left_stage = 'down'

        # Right leg logic
        if right_knee_y < right_hip_y - self.up_threshold:
            if self.right_stage == 'down':
                if now - self._last_time >= self.min_dt:
                    self.counter += 1
                    rep_done = True
                    self._last_time = now
            self.right_stage = 'up'
        else:
            self.right_stage = 'down'

        return rep_done, self.left_stage, self.right_stage, lm

    def draw_hud(self, frame, fps=0.0):
        """Draw a simple HUD: reps and optional FPS."""
        try:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (260,110), (0,0,0), -1)
            cv2.putText(frame, f"High Steps: {self.counter}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"L:{self.left_stage} R:{self.right_stage}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            cv2.putText(frame, f"{fps:.1f} FPS", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        except Exception:
            pass

    def process(self, frame_bgr):
        """
        Process a single BGR frame and return (processed_frame_bgr, count).
        - mirrors frame (like other processors)
        - runs MediaPipe pose
        - draws landmarks and HUD
        - updates internal counter/state
        """
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # draw landmarks if present
        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # update counter/state
        try:
            rep_done, l_stage, r_stage, lm = self.update_from_results(results)
        except Exception:
            rep_done, l_stage, r_stage, lm = False, self.left_stage, self.right_stage, None

        # draw HUD (fps unknown here - 0.0)
        self.draw_hud(frame, fps=0.0)

        return frame, self.counter


# module-level instance (import this in app.py)
highstep_counter = HighStepCounter()
