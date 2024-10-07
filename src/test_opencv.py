import cv2
import depth_pro
import numpy as np

# Load model and preprocessing transform
print("Loading model and preprocessing transform...")
model, transform = depth_pro.create_model_and_transforms()
print("Model loaded successfully!")
model.eval()

# Initialize video capture (use 0 for the default camera or provide a video path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame using the transform
    image = transform(frame_rgb)

    # Simulate focal length (this should be defined or derived from the camera parameters)
    f_px = 700  # Adjust the focal length based on your camera specifications

    # Run inference
    print("Running inference...")
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters

    # Normalize depth for visualization
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # Stack the original frame and depth map horizontally
    combined = np.hstack((frame, depth_colored))

    # Display the combined result
    cv2.imshow("Depth Estimation", combined)

    # Wait for a short period and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()