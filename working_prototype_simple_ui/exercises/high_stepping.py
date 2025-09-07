import numpy as np
import math
import mediapipe as mp

# MediaPipe Pose setup
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points in degrees.
    This function is a helper for various pose-related calculations.
    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    # Vectorized calculation for efficiency
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/math.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_high_steps_frame(landmarks, current_count, left_stage, right_stage):
    """
    Processes a single frame's landmarks to count high steps.

    Args:
        landmarks: The pose landmarks from MediaPipe.
        current_count: The current count of high steps.
        left_stage: The current stage of the left leg ('up' or 'down').
        right_stage: The current stage of the right leg ('up' or 'down').

    Returns:
        A tuple containing the updated count, left stage, and right stage.
    """
    # Get the y-coordinates of the hip and knee landmarks for both legs.
    # We use y-coordinates to determine if the knee is raised above the hip.
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    
    # Logic to count high steps for the left leg
    # A step is counted when the knee is raised above a certain threshold relative to the hip,
    # and the stage changes from 'down' to 'up'.
    if left_knee_y < left_hip_y - 0.1:
        if left_stage == 'down':
            current_count += 1
        left_stage = 'up'
    else:
        left_stage = 'down'

    # Logic to count high steps for the right leg
    if right_knee_y < right_hip_y - 0.1:
        if right_stage == 'down':
            current_count += 1
        right_stage = 'up'
    else:
        right_stage = 'down'
    
    return current_count, left_stage, right_stage
