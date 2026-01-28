import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import cv2 

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "final_model_SHADING.pth"
INPUT_IMAGE = "test_face.jpg"
OUTPUT_FINAL = "result_FINAL_POLISH.png"
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
# 3. THE "SURGICAL POLISH" PIPELINE (Layered + Masked Noise)
# ==========================================
def make_it_polished(tensor_output, output_path):
    print("✨ Applying Surgical Polish (Structure + Detail + Masked Noise)...")
    
    # Denormalize
    img = tensor_output.squeeze().cpu().detach().numpy()
    img = (img + 1) / 2.0 * 255.0  
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Upscale (Lanczos 1024px)
    img_hd = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

    # --- LAYER A: STRUCTURE (Bones) ---
    # High threshold (220) to keep ONLY jaw, eyes, and main outlines.
    layer_struct = cv2.GaussianBlur(img_hd, (5, 5), 0)
    _, layer_struct = cv2.threshold(layer_struct, 220, 255, cv2.THRESH_BINARY)

    # --- LAYER B: DETAIL (Texture) ---
    # Gamma 0.75 makes hair faint gray (pencil) instead of black (ink).
    gamma = 0.75
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    layer_detail = cv2.LUT(img_hd, table)
    # Soften details so teeth/wrinkles look shaded, not outlined.
    layer_detail = cv2.GaussianBlur(layer_detail, (3, 3), 0)

    # --- MERGE ---
    # Combine Strong Lines + Soft Details
    final_blend = np.minimum(layer_struct, layer_detail)

    # --- PAPER TEXTURE (Masked Noise) ---
    # Only add noise where lines exist (darker than 250). White paper stays white.
    stroke_mask = final_blend < 250
    noise = np.random.normal(0, 4, final_blend.shape).astype(int)

    final_int = final_blend.astype(np.int16)
    noise_layer = np.zeros_like(final_int)
    noise_layer[stroke_mask] = noise[stroke_mask]
    
    final_int = final_int - noise_layer
    final_img = np.clip(final_int, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, final_img)
    print(f"✅ Saved FINAL POLISH result to: {output_path}")

# ==========================================
# 4. RUN
# ==========================================
def run_test():
    print(f"🚀 Running Inference...")
    gen = SketchGenerator().to(DEVICE)
    try:
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        gen.eval()
    except:
        print("❌ Model not found!")
        return

    img = Image.open(INPUT_IMAGE).convert("RGB")
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_tensor = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): 
        output = gen(img_tensor)
        
    make_it_polished(output, OUTPUT_FINAL)

if __name__ == "__main__":
    run_test()