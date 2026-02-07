import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "final_model_ATTENTION.pth" 
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. MODEL ARCHITECTURE (Must match file)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.fc1 = nn.Conv2d(planes, planes // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(planes // 16, planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(nn.AdaptiveAvgPool2d(1)(x))))
        max_out = self.fc2(self.relu(self.fc1(nn.AdaptiveMaxPool2d(1)(x))))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

class SketchGenerator_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc1 = nn.Sequential(*list(vgg.children())[:4])   
        self.enc2 = nn.Sequential(*list(vgg.children())[4:14]) 
        self.enc3 = nn.Sequential(*list(vgg.children())[14:24])
        self.attention = CBAM(512)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.final = nn.Conv2d(64, 1, 3, 1, 1)
    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f3 = self.attention(f3)
        x = self.up1(f3)
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:])
        x = self.up2(torch.cat([x, f2_rs], 1))
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:])
        x = self.up3(torch.cat([x, f1_rs], 1))
        return torch.tanh(self.final(x))

# ==========================================
# 3. THE "HEAVY GRAPHITE" ENGINE
# ==========================================
def apply_heavy_graphite_style(img):
    """
    Forces the 'faded' output to become a confident pencil sketch.
    Simulates pressure by thickening dark lines.
    """
    # 1. UPSCALING (Crucial for crisp edges)
    # We go big so we can have fine grain control
    h, w = img.shape[:2]
    scale = 4 # 4x upscale (1024px)
    img_hd = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)

    # 2. SEPARATE LAYERS (The "Project Vibe" Logic)
    # Layer A: The "Ink" (Deep Blacks)
    # We threshold hard to find the confident lines
    _, ink_mask = cv2.threshold(img_hd, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Layer B: The "Ghost" (Faint Hair/Shading)
    # We use adaptive thresholding to find texture in the white areas
    ghost_layer = cv2.adaptiveThreshold(
        img_hd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # 3. PRESSURE SIMULATION (Morphology)
    # If it's "Ink" (confident), we Dilate (thicken) it slightly
    kernel_strong = np.ones((3,3), np.uint8)
    ink_thick = cv2.dilate(ink_mask, kernel_strong, iterations=1)
    
    # If it's "Ghost" (hair), we keep it thin but darken it
    # We merge them: Strong lines + Faint details
    merged = cv2.addWeighted(ink_thick, 0.8, ghost_layer, 0.4, 0)
    
    # 4. INVERT BACK TO WHITE PAPER
    sketch = 255 - merged
    
    # 5. NOISE TEXTURE (The "Paper Grain")
    # Add random noise only to the dark parts to break the "digital" look
    noise = np.random.normal(0, 15, sketch.shape).astype(np.uint8)
    sketch = cv2.subtract(sketch, noise) # Etch noise into graphite
    
    # 6. FINAL CONTRAST PUNCH
    # Crush the blacks one last time
    _, sketch = cv2.threshold(sketch, 200, 255, cv2.THRESH_TRUNC)
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    
    return sketch

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    # Load Model
    print(f"⏳ Loading Model from {MODEL_PATH}...")
    model = SketchGenerator_Attention().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print("❌ Model not found! Please check file name.")
        return

    # Load Image
    img_path = "test_face.jpg"
    if len(sys.argv) > 1: img_path = sys.argv[1]
    
    print(f"🎨 Processing {img_path}...")
    img = Image.open(img_path).convert("RGB")
    
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Inference
    with torch.no_grad():
        tens = tf(img).unsqueeze(0).to(DEVICE)
        out = model(tens)
        
    # Raw Output
    raw = out.squeeze().cpu().detach().numpy()
    raw = (raw + 1) / 2.0 * 255.0
    raw = np.clip(raw, 0, 255).astype(np.uint8)
    
    # Apply The Fix
    final_result = apply_heavy_graphite_style(raw)
    
    # Save
    cv2.imwrite("result_HEAVY_GRAPHITE.png", final_result)
    print("✅ Saved to 'result_HEAVY_GRAPHITE.png'")
    print("👉 Check this file. It should have THICK, DARK lines.")

if __name__ == "__main__":
    main()