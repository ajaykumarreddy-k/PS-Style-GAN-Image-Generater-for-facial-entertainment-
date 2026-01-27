import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# 1. CONFIG
# ==========================================
class Config:
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "final_model.pth"   # This file was just created by your training script
    INPUT_IMAGE = "test_face.jpg"    # <--- RENAME YOUR SELFIE TO THIS
    OUTPUT_NAME = "result_sketch.png"

# ==========================================
# 2. MODEL DEFINITION (Must match training exactly)
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
        if f2.shape[2:] != x.shape[2:]: f2 = nn.functional.interpolate(f2, size=x.shape[2:])
        x = self.up2(torch.cat([x, f2], 1))
        if f1.shape[2:] != x.shape[2:]: f1 = nn.functional.interpolate(f1, size=x.shape[2:])
        x = self.up3(torch.cat([x, f1], 1))
        return torch.tanh(self.final(x))

# ==========================================
# 3. RUN
# ==========================================
def run():
    if not os.path.exists(Config.MODEL_PATH):
        print("❌ Error: final_model.pth not found. Did training finish?")
        return
    if not os.path.exists(Config.INPUT_IMAGE):
        print(f"❌ Error: Put a photo named '{Config.INPUT_IMAGE}' in this folder first!")
        return

    print("🚀 Loading Model...")
    model = SketchGenerator().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    img = Image.open(Config.INPUT_IMAGE).convert("RGB")
    original_size = img.size
    
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    with torch.no_grad():
        tensor = tf(img).unsqueeze(0).to(Config.DEVICE)
        out = model(tensor)
        
    # Convert back to image
    out = (out + 1) / 2.0
    pil_out = transforms.ToPILImage()(out.squeeze().cpu())
    pil_out = pil_out.resize(original_size, Image.BICUBIC)
    
    pil_out.save(Config.OUTPUT_NAME)
    print(f"✨ Saved sketch to {Config.OUTPUT_NAME}")
    
    # Display result
    # Display result (Modified to SAVE instead of SHOW)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
    plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(pil_out, cmap='gray'); plt.title("PS Style GAN")
    plt.axis("off")
    
    # CHANGED: Save the comparison image instead of crashing on .show()
    plt.savefig("comparison_result.png")
    print("📸 Saved side-by-side comparison to 'comparison_result.png'")

if __name__ == "__main__":
    run()