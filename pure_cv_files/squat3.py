import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

# ---------- Helpers ----------
def get_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def moving_average(buf, val, maxlen=5):
    buf.append(val)
    return sum(buf) / len(buf)

def draw_hud(frame, count, phase, angle, fps):
    cv2.rectangle(frame, (10, 10), (260, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Squats: {count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Phase: {phase}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
    cv2.putText(frame, f"Knee: {int(angle)} deg", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ---------- Pose + Counter ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class SquatCounter:
    def __init__(self, up_thresh=170, down_thresh=90, min_dt=0.8, smooth_len=7):
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.min_dt = min_dt
        self.buf = deque(maxlen=smooth_len)
        self.state = "up"
        self.count = 0
        self.last_time = 0.0

    def _get_points(self, results):
        if results.pose_landmarks is None:
            return None
        lm = results.pose_landmarks.landmark
        L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        L_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
        R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
        R_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

        left = ((lm[L_HIP].x, lm[L_HIP].y),
                (lm[L_KNEE].x, lm[L_KNEE].y),
                (lm[L_ANKLE].x, lm[L_ANKLE].y))
        right = ((lm[R_HIP].x, lm[R_HIP].y),
                 (lm[R_KNEE].x, lm[R_KNEE].y),
                 (lm[R_ANKLE].x, lm[R_ANKLE].y))
        return left, right

    def update(self, results):
        pts = self._get_points(results)
        if pts is None:
            return 180, self.state, False
        (l_hip,l_knee,l_ankle),(r_hip,r_knee,r_ankle) = pts
        angle_l = get_angle(l_hip,l_knee,l_ankle)
        angle_r = get_angle(r_hip,r_knee,r_ankle)
        angle = min(angle_l, angle_r)
        angle = moving_average(self.buf, angle)

        rep_done = False
        now = time.time()

        if self.state=="up":
            if angle < self.down_thresh:
                self.state="down"
        elif self.state=="down":
            if angle > self.up_thresh:
                if now-self.last_time >= self.min_dt:
                    self.count += 1
                    rep_done=True
                    self.last_time=now
                self.state="up"
        return angle, self.state, rep_done

# ---------- Main ----------
def main():
    cap=cv2.VideoCapture(0)
    cap.set(3,1280); cap.set(4,720)
    counter=SquatCounter()

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        prev=time.time()
        fps=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results=pose.process(rgb)

            # Draw pose landmarks (simpler style)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
                )

            angle,phase,_=counter.update(results)
            now=time.time(); fps=1/(now-prev); prev=now
            draw_hud(frame,counter.count,phase,angle,fps)

            cv2.putText(frame,"Press 'r' to reset, 'q' to quit",
                        (20,180),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.imshow("Squat Counter",frame)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==ord('r'):
                counter.count=0; counter.state="up"; counter.buf.clear(); counter.last_time=0
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
