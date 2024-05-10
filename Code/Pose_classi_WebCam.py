import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Create a VideoCapture object for webcam feed
webcam = cv2.VideoCapture(0)

# If webcam fails to open, raise an error
if not webcam.isOpened():
    raise IOError("Cannot open webcam")

# Standard size for dataset 1280*720
webcam_width = 1920
webcam_height = 1080

# Set webcam resolution and FPS
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
webcam.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second to 30
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set MJPG as video codec

# Load the entire model
#Deeper FC model
class Deep_FC_Model(nn.Module):
    def __init__(self):
        super(Deep_FC_Model, self).__init__()
        self.fc1 = nn.Linear(33 * 3, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 8)       # Output layer with 8 units for 8 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)         # Flatten the tensor
        x = F.relu(self.fc1(x))           # Apply ReLU activation function
        x = self.fc2(x)                   # No activation, if using BCEWithLogitsLoss
        return x

# Initialize the model and hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Deep_FC_Model().to(device)

# Replace this with your actual model file path
model_file_path = '/Users/abhishekchothani/01 Study/04 EE CS/04.03 Artificial Intelligence/02 Code/00 Mirai Code/Mirai3_Drone_Gesture_Control/Models/model_weights.pth'

# Attempt to load the model file
try:
    model_state_dict = torch.load(model_file_path)
    model.load_state_dict(model_state_dict)
    print("Model file loaded successfully.")
except Exception as e:
    print(f"Error loading model file: {e}")

model.eval()

class_names = [
    'backward', 'do nothing', 'forward', 'land', 
    'left', 'right', 'takeoff', 'up' 
]

def process_landmarks_for_model(landmarks):
    # Assuming landmarks is a list of (33, 3) shape
    landmarks_array = np.array(landmarks, dtype=np.float32)  # Explicitly specify dtype as float32
    # Reshape to (1, 33, 3) - adding an extra dimension for batch size
    return landmarks_array.reshape(1, 33, 3)

# Class for handling pose detection
class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=static_image_mode, 
                                               model_complexity=model_complexity, 
                                               min_detection_confidence=min_detection_confidence, 
                                               min_tracking_confidence=min_tracking_confidence)
        self.landmark_drawer = mp.solutions.drawing_utils

    # Method to process a frame, detect pose and draw landmarks
    def detect_and_draw_landmarks(self, frame):
        pose_positions = []

        # Convert BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find pose landmarks
        pose_landmarks_result = self.pose_detector.process(rgb_frame)

        # If pose landmarks are detected, draw them on the frame
        if pose_landmarks_result.pose_landmarks:
            self.landmark_drawer.draw_landmarks(frame, pose_landmarks_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks_coordinates = [(landmark.x, landmark.y, landmark.z) for landmark in pose_landmarks_result.pose_landmarks.landmark]
            pose_positions.append(landmarks_coordinates)
        return pose_positions, frame

# Create a PoseDetector object
pose_detector = PoseDetector()

# Start webcam feed and continue until 'q' is pressed
while True:
    # Capture frame from webcam
    ret, frame = webcam.read()

    # Get pose landmarks and lable frame from the image from the image
    pose_landmarks, pose_frame = pose_detector.detect_and_draw_landmarks(frame)

    if pose_landmarks:
        # Process landmarks for your model
        processed_landmarks = process_landmarks_for_model(pose_landmarks[0])
        input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            predictions = model(input_tensor.to(device))

        # Process the predictions
        predicted_class = torch.argmax(predictions, dim=1)
        predicted_label = class_names[predicted_class.item()]

        # Display the prediction on the frame
        cv2.putText(pose_frame, f'Predicted Pose: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Verify that frame has been successfully read
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Pose Detection Webcam Feed', pose_frame)

    # If 'q' is pressed on the keyboard, break the loop and end the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the webcam object
webcam.release()

# Destroy all the windows
cv2.destroyAllWindows()
