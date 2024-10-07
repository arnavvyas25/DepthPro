from PIL import Image
import depth_pro
import matplotlib.pyplot as plt
import cv2

# Load model and preprocessing transform
print("Loading model and preprocessing transform...")
model, transform = depth_pro.create_model_and_transforms()
print("Model loaded successfully!")
model.eval()

# Initialize video capture (use 0 for the default camera or provide a video path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess the frame using the transform
    image = transform(pil_image)

    # Simulate focal length (this should be defined or derived from the camera parameters)
    f_px = 700  # Adjust the focal length based on your camera specifications

    # Run inference
    print("Running inference...")
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters

    # Display the original image and depth map
    plt.figure(figsize=(10, 5))

    # Display the original frame
    plt.subplot(1, 2, 1)
    plt.imshow(pil_image)
    plt.title("Image")
    plt.axis("off")

    # Display the depth map
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="inferno")
    plt.title("Depth map")
    plt.axis("off")

    plt.pause(0.001)  # Use a small pause to allow the plot to refresh in real time

    # Clear the previous plot
    plt.clf()

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
plt.close()