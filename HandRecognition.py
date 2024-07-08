import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dictionary to map landmark points to specific ASL gestures
gesture_dict = {
    "Fist": [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)],  # All fingertips near their respective PIP joints
    "Open Hand": [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)],  # All fingertips far from their respective PIP joints
    "Thumbs Up": [(4, 2), (8, 7), (12, 11), (16, 15), (20, 19)],  # Thumb tip far from thumb MCP, other fingertips near their PIP joints
    "I Love You": [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)],  # Combination of fist, open hand and thumb position
    # Add more gestures here with the corresponding landmark indices
}

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

# Function to check if landmarks match the gesture
def match_gesture(landmarks):
    for gesture, points in gesture_dict.items():
        match = True
        for (start, end) in points:
            start_point = landmarks.landmark[start]
            end_point = landmarks.landmark[end]
            distance = euclidean_distance(start_point, end_point)
            
            if gesture == "Fist" and distance > 0.05:  # Adjust threshold as necessary
                match = False
                break
            elif gesture == "Open Hand" and distance < 0.2:  # Adjust threshold as necessary
                match = False
                break
            elif gesture == "Thumbs Up" and start == 4 and distance < 0.2:  # Thumb should be extended
                match = False
                break
            elif gesture == "Thumbs Up" and start != 4 and distance > 0.1:  # Other fingers should be near PIP
                match = False
                break
            # Add more conditions for other gestures here

        if match:
            return gesture
    return None

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Translate landmarks to gesture
            gesture = match_gesture(hand_landmarks)
            
            # Display the gesture on the image
            if gesture:
                cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('ASL Translator', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()