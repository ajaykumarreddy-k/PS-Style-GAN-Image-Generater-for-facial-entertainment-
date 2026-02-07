import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Model Architecture (same as hybrid_sketch_app.py)
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
        for p in self.parameters(): p.requires_grad = False
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
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:]) if f2.shape[2:] != x.shape[2:] else f2
        x = self.up2(torch.cat([x, f2_rs], 1))
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:]) if f1.shape[2:] != x.shape[2:] else f1
        x = self.up3(torch.cat([x, f1_rs], 1))
        return torch.tanh(self.final(x))

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "final_model_SHADING.pth"

print(f"Loading model from {model_path}...")
netG = SketchGenerator().to(device)
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    netG.load_state_dict(checkpoint['model_state_dict'])
else:
    netG.load_state_dict(checkpoint)
netG.eval()
print(f"✅ Model loaded on {device}")

# Load Image
print("\nProcessing test_face.jpg...")
image = Image.open("test_face.jpg").convert("RGB")

# STEP 1: CV2 Line Art
img_np = np.array(image)
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
inverted = 255 - gray
k_size = 43  # blur_k=21 → k_size=43
blurred = cv2.GaussianBlur(inverted, (k_size, k_size), 0)
sketch_cv2 = cv2.divide(gray, 255 - blurred, scale=256)

# STEP 2: AI Shading
sketch_rgb = Image.fromarray(sketch_cv2).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input_tensor = transform(sketch_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    generated = netG(input_tensor)
    generated = generated.squeeze().cpu().detach().numpy()
    generated = (generated * 0.5 + 0.5) * 255.0
    generated = np.clip(generated, 0, 255).astype(np.uint8)
    
    if generated.shape != sketch_cv2.shape:
        generated = cv2.resize(generated, (sketch_cv2.shape[1], sketch_cv2.shape[0]), 
                             interpolation=cv2.INTER_LANCZOS4)

# STEP 3: Create multiple blends
print("\nGenerating results with different blend ratios...")

# Save original
cv2.imwrite("hybrid_result_ORIGINAL.jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# Save pure CV2
cv2.imwrite("hybrid_result_CV2_ONLY.png", sketch_cv2)
print("✅ Saved: hybrid_result_CV2_ONLY.png (Pure CV2 line art)")

# Save pure AI
cv2.imwrite("hybrid_result_AI_ONLY.png", generated)
print("✅ Saved: hybrid_result_AI_ONLY.png (Pure GAN shading)")

# Save blends at different strengths
for strength in [25, 50, 75, 100]:
    alpha = strength / 100.0
    blended = cv2.addWeighted(generated, alpha, sketch_cv2, 1 - alpha, 0)
    filename = f"hybrid_result_BLEND_{strength}.png"
    cv2.imwrite(filename, blended)
    print(f"✅ Saved: {filename} ({strength}% AI strength)")

print("\n🎨 All results saved! Check the files:")
print("   - hybrid_result_CV2_ONLY.png (mathematical precision)")
print("   - hybrid_result_BLEND_75.png (recommended blend)")
print("   - hybrid_result_AI_ONLY.png (full artistic shading)")
