import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

# Project modules
from model import UNET

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model ---
checkpoint_path = os.path.join("run", "train35", "weights", "best_model.pth.tar")
model = UNET(in_channels=1, out_channels=1)
ckpt = torch.load(checkpoint_path, map_location=device)
state = ckpt.get("state_dict", ckpt)
model.load_state_dict(state)
model.to(device)
model.eval()

# Default threshold
DEFAULT_THRESHOLD = 0.35

# Prediction function
def predict(image: Image.Image, threshold: float):
    # Convert to grayscale numpy array
    img = np.array(image.convert("L"), dtype=np.uint8)
    # Resize to model input size
    img_resized = cv2.resize(img, (256, 256))
    # Normalize to [0,1]
    img_norm = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Binarize with threshold
    mask = (prob_map > threshold).astype(np.uint8) * 255

    # Create RGB overlay
    overlay = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    overlay[mask == 255] = [255, 0, 0]
    # Resize output to 512×512
    overlay = cv2.resize(overlay, (512, 512))
    return overlay

# Gradio interface setup
image_input = gr.Image(type="pil", label="Upload CT Slice")
threshold_input = gr.Slider(minimum=0.1, maximum=0.9, value=DEFAULT_THRESHOLD, label="Probability Threshold")
overlay_output = gr.Image(type="numpy", label="Segmentation Overlay")

tool = gr.Interface(
    fn=predict,
    inputs=[image_input, threshold_input],
    outputs=[overlay_output],
    title="Lung Nodule Segmentation",
    description=(
        "Upload a single axial CT slice (PNG/JPG). "
        "The model returns a red overlay where nodules are predicted. "
        "Adjust the probability threshold as needed."
    ),
    flagging_mode="never"
)

if __name__ == "__main__":
    tool.launch(server_name="127.0.0.1", server_port=7860)