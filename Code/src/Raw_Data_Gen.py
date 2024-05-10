import cv2
import time

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

# Initial time
start_time = time.time()
interval = 2
flash_duration = 0.1 

try:
    frame_count = 0
    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()

        # If frame reading fails, break the loop
        if not ret:
            break

        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = interval - elapsed_time

        # Check if it's time to save the frame
        if remaining_time <= 0:
            # Save the frame
            frame_name = f"webcam_frame_{frame_count}.jpg"
            cv2.imwrite(frame_name, frame)
            print(f"Saved {frame_name}")

            # Flash the screen white
            frame[:] = (255, 255, 255)
            cv2.imshow('Webcam Feed', frame)
            cv2.waitKey(int(flash_duration * 1000))  # Wait for the duration of the flash

            # Update the start time and frame count
            start_time = time.time()
            frame_count += 1
            continue

        # Display the countdown timer and frame number on the frame
        countdown_text = f"Frame: {frame_count} | Saving in: {int(remaining_time)}s"
        cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and destroy all windows
    webcam.release()
    cv2.destroyAllWindows()
