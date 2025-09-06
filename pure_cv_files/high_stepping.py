import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  
    
    ab = a - b
    cb = c - b
    
    radians = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    angle = np.degrees(radians)
    
    return angle

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    counter = 0
    left_stage = 'down'
    right_stage = 'down'

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                
                right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                
                if left_knee_y < left_hip_y - 0.1:
                    if left_stage == 'down':
                        counter += 1
                    left_stage = 'up'
                else:
                    left_stage = 'down'

                if right_knee_y < right_hip_y - 0.1:
                    if right_stage == 'down':
                        counter += 1
                    right_stage = 'up'
                else:
                    right_stage = 'down'
                    
            except:
                pass
                
            cv2.putText(image, f"HIGH STEPS: {counter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"LEFT STAGE: {left_stage}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"RIGHT STAGE: {right_stage}", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                                     
            cv2.imshow('High Step Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
