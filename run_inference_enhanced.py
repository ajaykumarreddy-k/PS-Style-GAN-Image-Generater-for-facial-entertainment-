import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageOps
from torchvision.utils import save_image
import os
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "final_model_SHADING.pth"  # Your risky model
INPUT_IMAGE = "test_face.jpg"
OUTPUT_RAW = "result_raw.png"
OUTPUT_FINAL = "result_studio_finish.png"
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
# 3. POST-PROCESSING MAGIC
# ==========================================
def develop_sketch(image_path, output_path):
    """Turns a faint raw GAN output into a bold pencil sketch"""
    img = Image.open(image_path).convert("L") # Grayscale
    
    # 1. Boost Contrast (Make darks darker)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Double the contrast
    
    # 2. Sharpen (Define the lines)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)

    # 3. Clean White Noise (Thresholding)
    # Any pixel brighter than 200 becomes pure white (removes gray dust)
    img_np = np.array(img)
    img_np = np.where(img_np > 220, 255, img_np) 
    img = Image.fromarray(img_np.astype(np.uint8))
    
    img.save(output_path)
    print(f"✨ Developed Studio Version saved to: {output_path}")

# ==========================================
# 4. RUN
# ==========================================
def run_test():
    print(f"🚀 Running Enhanced Inference...")
    gen = SketchGenerator().to(DEVICE)
    try:
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        gen.eval()
    except: return

    img = Image.open(INPUT_IMAGE).convert("RGB")
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_tensor = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): output = gen(img_tensor)
    save_image(output, OUTPUT_RAW, normalize=True)
    
    # APPLY THE FIX
    develop_sketch(OUTPUT_RAW, OUTPUT_FINAL)

if __name__ == "__main__":
    run_test()