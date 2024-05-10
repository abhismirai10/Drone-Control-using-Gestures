import cv2
import mediapipe as mp
import pandas as pd

# Class for handling pose detection
class PoseDetector:
    def __init__(self, static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
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

# DataFrame to store all landmarks for all images
all_landmarks = pd.DataFrame()

#number of datapoints
n=200

for i in range(n):  # Loop over n images

    # Load an image
    image_path = f'Raw Dataset/up100/up{i}.jpg'
    image = cv2.imread(image_path)

    # Check if image is loaded properly
    if image is None:
        raise IOError(f"Cannot load image {image_path}")

    # Get pose landmarks and lable frame from the image from the image
    pose_landmarks, pose_frame = pose_detector.detect_and_draw_landmarks(image)

    # Check if landmarks were detected
    if pose_landmarks:
        # Create a DataFrame for each landmark with frame number and point number
        landmarks_df = pd.DataFrame({
            'frame': i,
            'point': range(len(pose_landmarks[0])),
            'x': [landmark[0] for landmark in pose_landmarks[0]],
            'y': [landmark[1] for landmark in pose_landmarks[0]],
            'z': [landmark[2] for landmark in pose_landmarks[0]]
        })

        # Append landmarks to the all_landmarks DataFrame
        all_landmarks = pd.concat([all_landmarks, landmarks_df], ignore_index=True)

        # Mark each pose position in blue
        for position in pose_landmarks[0]:
            x, y = int(position[0] * image.shape[1]), int(position[1] * image.shape[0])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    # Save the image with drawn landmarks
    #cv2.imwrite(f'Raw Dataset/2a/webcam_frame_{i}.jpg', image)

    # # Display the resulting frame
    # cv2.imshow('Pose Detection', image)
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()

# Save landmarks data to CSV
all_landmarks.to_csv('Fine Tune Dataset/up200.csv', index=False)
print("Processing complete. Images and CSV file saved.")
