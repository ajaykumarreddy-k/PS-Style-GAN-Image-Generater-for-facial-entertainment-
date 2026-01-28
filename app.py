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
MODEL_PATH = "final_model_SHADING.pth"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. MODEL ARCHITECTURE
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
# 3. INITIALIZE MODEL
# ==========================================
print("⏳ Loading Model...")
model = SketchGenerator().to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model Loaded Successfully!")
else:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found.")

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
def convert_to_sketch(input_image):
    if input_image is None:
        return None
        
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
    
    # 4. Polish Logic
    img_hd = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    
    # Structure
    layer_struct = cv2.GaussianBlur(img_hd, (5, 5), 0)
    _, layer_struct = cv2.threshold(layer_struct, 220, 255, cv2.THRESH_BINARY)
    
    # Detail
    gamma = 0.75
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    layer_detail = cv2.LUT(img_hd, table)
    layer_detail = cv2.GaussianBlur(layer_detail, (3, 3), 0)
    
    # Merge
    final_blend = np.minimum(layer_struct, layer_detail)
    
    # Masked Noise
    stroke_mask = final_blend < 250
    noise = np.random.normal(0, 4, final_blend.shape).astype(int)
    final_int = final_blend.astype(np.int16)
    noise_layer = np.zeros_like(final_int)
    noise_layer[stroke_mask] = noise[stroke_mask]
    
    final_img = np.clip(final_int - noise_layer, 0, 255).astype(np.uint8)

    return final_img

# ==========================================
# 5. PROFESSIONAL GRADIO UI (BLOCKS)
# ==========================================
custom_css = """
#container {
    max-width: 1200px;
    margin: auto;
    padding-top: 20px;
}
#header {
    text-align: center;
    margin-bottom: 30px;
}
#header h1 {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 10px;
}
#header p {
    font-size: 1.1em;
    color: #666;
}
.generate-btn {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
    border: none !important;
    color: white !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    with gr.Column(elem_id="container"):
        
        # Header
        with gr.Column(elem_id="header"):
            gr.Markdown("# 🎨 Project Vibe: AI Sketch Studio")
            gr.Markdown("Transform portraits into **professional graphite sketches** using Layered GAN Synthesis.")

        # Main Interface
        with gr.Row():
            # Left: Input
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Portrait", height=450)
                run_btn = gr.Button("✨ Generate Sketch", elem_classes="generate-btn", variant="primary")
            
            # Right: Output
            with gr.Column():
                output_img = gr.Image(type="numpy", label="Graphite Result", height=450)

        # Examples Footer
        gr.Markdown("### 🔍 Try Examples")
        gr.Examples(
            examples=["test_face.jpg"], # Ensure this file exists in your folder!
            inputs=input_img,
            outputs=output_img,
            fn=convert_to_sketch,
            cache_examples=True,
        )

    # Event Binding
    run_btn.click(fn=convert_to_sketch, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(share=True)