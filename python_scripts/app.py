import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🏆 REVERTING TO THE CHAMPION MODEL
MODEL_PATH = "final_model_SHADING.pth" 
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. ARCHITECTURE (The Proven Champion)
# ==========================================
class SafeFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + (noise * self.noise_scale)

class SketchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc1 = nn.Sequential(*list(vgg.children())[:4])   
        self.enc2 = nn.Sequential(*list(vgg.children())[4:14]) 
        self.enc3 = nn.Sequential(*list(vgg.children())[14:24])
        self.fuse = SafeFusionModule(512)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.final = nn.Conv2d(64, 1, 3, 1, 1)
    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        x = self.up1(self.fuse(f3))
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:]) if f2.shape[2:]!=x.shape[2:] else f2
        x = self.up2(torch.cat([x, f2_rs], 1))
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:]) if f1.shape[2:]!=x.shape[2:] else f1
        x = self.up3(torch.cat([x, f1_rs], 1))
        return torch.tanh(self.final(x))

# ==========================================
# 3. INITIALIZE
# ==========================================
print(f"⏳ Loading Champion Model: {MODEL_PATH}...")
model = SketchGenerator().to(DEVICE)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
else:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found.")

# ==========================================
# 4. INFERENCE (The "Final Polish" Logic)
# ==========================================
def convert_to_sketch(input_image):
    if input_image is None: return None
    
    # 1. Preprocess
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = tf(input_image).unsqueeze(0).to(DEVICE)

    # 2. Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # 3. Denormalize
    img = output.squeeze().cpu().detach().numpy()
    img = (img + 1) / 2.0 * 255.0  
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 4. LAYERED SYNTHESIS (Restoring the method you liked)
    img_hd = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

    # Layer A: Structure (Bold Lines)
    layer_struct = cv2.GaussianBlur(img_hd, (5, 5), 0)
    _, layer_struct = cv2.threshold(layer_struct, 220, 255, cv2.THRESH_BINARY)

    # Layer B: Detail (Soft Texture)
    # We use Gamma Correction to pull out the faint hair details
    gamma = 0.75
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    layer_detail = cv2.LUT(img_hd, table)
    layer_detail = cv2.GaussianBlur(layer_detail, (3, 3), 0)

    # Merge: Combine Bold Structure with Soft Detail
    final_blend = np.minimum(layer_struct, layer_detail)

    # 5. DENSITY BOOST (The Safe "Darken" Fix)
    # Instead of noise, we just multiply the image by itself to deepen the blacks.
    # This respects the existing lines and just makes them "inkier".
    img_f = final_blend.astype(float) / 255.0
    density = img_f * img_f # Multiply Blend
    final_img = (density * 255).astype(np.uint8)
    
    return final_img

# ==========================================
# 5. UI
# ==========================================
custom_css = """
.generate-btn { background: linear-gradient(90deg, #222 0%, #555 100%) !important; color: white !important; font-weight: bold; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✏️ Project Vibe: Final Studio Edition")
    gr.Markdown("Powered by the **Champion Model** and **Layered Synthesis**.")
    
    with gr.Row():
        input_img = gr.Image(type="pil", label="Portrait")
        output_img = gr.Image(type="numpy", label="Sketch Result")
    
    btn = gr.Button("Generate Sketch", elem_classes="generate-btn")
    btn.click(convert_to_sketch, input_img, output_img)

if __name__ == "__main__":
    demo.launch(share=True)