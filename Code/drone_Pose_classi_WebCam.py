import time
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# ----- Model and Pose Detection Setup -----
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

# Load model weights
model_file_path = '/home/abhishek/03 New Things/Code/Mirai3_0/model_weights.pth'
try:
    model.load_state_dict(torch.load(model_file_path))
    print("Model file loaded successfully.")
except Exception as e:
    print(f"Error loading model file: {e}")
model.eval()

class_names = ['backward', 'do nothing', 'forward', 'land', 'left', 'right', 'takeoff', 'up']

# Function to process landmarks for model input
def process_landmarks_for_model(landmarks):
    landmarks_array = np.array(landmarks, dtype=np.float32)
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

pose_detector = PoseDetector()


# ----- Webcam Setup -----
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise IOError("Cannot open webcam")
# Set webcam properties...
# Standard size for dataset 1280*720
webcam_width = 1920
webcam_height = 1080

# Set webcam resolution and FPS
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
webcam.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second to 30
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set MJPG as video codec


# ----- Drone Control Functions -----
# Function to connect to the drone and check its status
async def connect_and_check_drone():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Checking drone's global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    return drone

# Global variables for current position
current_north = 0.0
current_east = 0.0
current_down = 0.0

async def send_drone_command(drone, pose):
    global current_north, current_east, current_down

    if pose == 'takeoff':
        print("-- Arming")
        await drone.action.arm()
        print("-- Taking off")
        current_down = 5.0

    elif pose == 'land':
        print("-- Landing")
        current_down = 0.0

    elif pose == 'do nothing':
        print("-- hold")

    elif pose == 'up':
        print("-- Arming")
        await drone.action.arm()

        print("-- Moving up")
        current_down -= 5.0

    elif pose == 'forward':
        print("-- Moving forward")
        current_north += 5.0

    elif pose == 'backward':
        print("-- Moving backward")
        current_north -= 5.0

    elif pose == 'left':
        print("-- Moving left")
        current_east -= 5.0

    elif pose == 'right':
        print("-- Moving right")
        current_east += 5.0

    # Apply the position change
    try:
        await drone.offboard.set_position_ned(PositionNedYaw(current_north, current_east, current_down, 0.0))
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard error: {error}")
        return

# ----- Main Program -----
async def main():
    drone = await connect_and_check_drone()
    last_pose = None
    pose_start_time = None
  
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        pose_landmarks, pose_frame = pose_detector.detect_and_draw_landmarks(frame)
        current_pose = None
        
        if pose_landmarks:
            # Process landmarks for your model
            processed_landmarks = process_landmarks_for_model(pose_landmarks[0])
            input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)

            # Get predictions
            with torch.no_grad():
                predictions = model(input_tensor.to(device))

            # Process the predictions
            predicted_class = torch.argmax(predictions, dim=1)
            current_pose = class_names[predicted_class.item()]

            # Display the prediction on the frame
            cv2.putText(pose_frame, f'Predicted Pose: {current_pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Check if the pose has changed
        if current_pose != last_pose:
            last_pose = current_pose
            pose_start_time = time.time()

        # Check if the pose has been held for 2 seconds
        if current_pose == last_pose and time.time() - pose_start_time >= 2:
            # Send command to the drone
            await send_drone_command(drone, current_pose)
            last_pose = None

        cv2.imshow('Pose Detection Webcam Feed', pose_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())