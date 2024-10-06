from PIL import Image
import depth_pro

image_path = "path/to/your/image.jpg"

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

# for a depth-based dataset
boundary_f1 = SI_boundary_F1(predicted_depth, target_depth)

# output depth as image
depth_image = depth_pro.depth_to_image(depth)
depth_image.save("path/to/your/output/depth_image.jpg")