from PIL import Image
import depth_pro
import matplotlib.pyplot as plt

image_path = "src/input_images/IMG_4272.jpg"

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

# plot depth map and image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path))
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap="inferno")
plt.title("Depth map")
plt.axis("off")

plt.show()