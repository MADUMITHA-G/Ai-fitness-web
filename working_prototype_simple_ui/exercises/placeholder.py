# processors/placeholder.py
import cv2
import mediapipe as mp

class PlaceholderProcessor:
    """
    Simple placeholder processor that draws MediaPipe pose landmarks on an input BGR frame.

    Usage:
        proc = PlaceholderProcessor()
        processed_frame, count = proc.process(frame_bgr)

    The processor intentionally returns count=0 (placeholder) so it fits into your
    existing Flask endpoints that expect (processed_frame, count).
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """
        Process a single BGR frame and draw pose landmarks.

        Args:
            frame_bgr: numpy.ndarray BGR image (as returned by cv2.imdecode)

        Returns:
            (processed_bgr, count)
            processed_bgr: same frame with landmarks drawn
            count: always 0 (placeholder)
        """
        # Mirror the frame so it feels natural to the user
        frame = cv2.flip(frame_bgr, 1)
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # Draw landmarks if detected
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
            )

        # Placeholder returns 0 so it can be used like other processors
        return frame, 0

    def reset(self):
        """Placeholder reset (no internal state)."""
        return

    def close(self):
        """Release resources if needed."""
        try:
            self.pose.close()
        except Exception:
            pass
