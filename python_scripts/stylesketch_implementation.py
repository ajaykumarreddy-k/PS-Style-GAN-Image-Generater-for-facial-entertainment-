import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import random
import zipfile
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    IMG_SIZE = 256
    BATCH_SIZE = 2
    EPOCHS = 50       # Reduced for quick results
    LR = 0.0001       # Slower learning rate for stability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ZIP_NAME = "SKSF-A.zip"
    EXTRACT_DIR = "./dataset_extracted"
    OUTPUT_DIR = "./output"
    
    LAMBDA_L1 = 100
    LAMBDA_MASK = 10

# ==========================================
# 2. DATASET (Robust)
# ==========================================
def setup_dataset():
    if not os.path.exists(Config.EXTRACT_DIR):
        if os.path.exists(Config.ZIP_NAME):
            with zipfile.ZipFile(Config.ZIP_NAME, 'r') as z:
                z.extractall(Config.EXTRACT_DIR)

    # Find 'Photo' folder (Case insensitive)
    possible_roots = glob.glob(f"{Config.EXTRACT_DIR}/**/Photo", recursive=True)
    if not possible_roots:
        possible_roots = glob.glob(f"{Config.EXTRACT_DIR}/**/photo", recursive=True)
        
    if not possible_roots:
        print("❌ CRITICAL: Could not find 'Photo' folder.")
        return None, None
        
    root_data_dir = os.path.dirname(possible_roots[0])
    csv_files = glob.glob(f"{root_data_dir}/*.csv")
    if not csv_files: csv_files = glob.glob(f"{Config.EXTRACT_DIR}/**/*.csv", recursive=True)

    if not csv_files:
        print("❌ CRITICAL: CSV missing.")
        return None, None
    
    return csv_files[0], root_data_dir

class SmartDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        try:
            self.df = pd.read_csv(csv_file)
        except:
            self.df = pd.DataFrame()
            return

        self.photo_map = self._scan_folder("Photo")
        self.mask_map = self._scan_folder("mask")
        
        self.style_maps = {}
        for i in range(1, 8):
            s_map = self._scan_folder(f"style{i}")
            if s_map: self.style_maps[i] = s_map
        
        self.id_col = -1
        for col in range(len(self.df.columns)):
            val = str(self.df.iloc[0, col]).strip()
            if val in self.photo_map:
                self.id_col = col
                break
        if self.id_col == -1: self.id_col = 0

    def _scan_folder(self, folder_name):
        path = os.path.join(self.root_dir, folder_name)
        file_map = {}
        if not os.path.exists(path): return file_map
        for f in os.listdir(path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_map[os.path.splitext(f)[0]] = f
        return file_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0: return None
        file_id = str(self.df.iloc[idx, self.id_col]).strip()
        
        if file_id not in self.photo_map: return self.__getitem__((idx + 1) % len(self.df))
            
        photo_path = os.path.join(self.root_dir, "Photo", self.photo_map[file_id])
        
        mask_name = self.mask_map.get(file_id)
        if mask_name: mask_path = os.path.join(self.root_dir, "mask", mask_name)
        else: mask_path = None
             
        valid_styles = list(self.style_maps.keys())
        if not valid_styles: return self.__getitem__((idx + 1) % len(self.df))
        style_path = os.path.join(self.root_dir, f"style{random.choice(valid_styles)}", self.style_maps[valid_styles[0]].get(file_id, ""))

        try:
            photo = Image.open(photo_path).convert("RGB")
            sketch = Image.open(style_path).convert("L")
            if mask_path: mask = Image.open(mask_path).convert("L")
            else: mask = Image.new("L", photo.size, 255)

            if self.transform:
                photo = self.transform(photo)
                mask = self.transform(mask)
                sketch = self.transform(sketch)
            return photo, mask, sketch
        except:
             return self.__getitem__((idx + 1) % len(self.df))

# ==========================================
# 3. STABILIZED ARCHITECTURE
# ==========================================

class SafeFusionModule(nn.Module):
    """
    STABLE NOVELTY: StyleGAN-inspired Noise Injection.
    Instead of dividing by variance (dangerous), we just add learned noise.
    """
    def __init__(self, channels):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1)) # Start with 0 noise
        
    def forward(self, x):
        # Generate random noise matching input shape
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        # Add noise scaled by the learned parameter
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
        
        # Standard U-Net Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.final = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        x = self.up1(self.fuse(f3)) # Apply fusion
        
        # Resize and skip connection
        if f2.shape[2:] != x.shape[2:]: f2 = nn.functional.interpolate(f2, size=x.shape[2:])
        x = self.up2(torch.cat([x, f2], 1))
        
        if f1.shape[2:] != x.shape[2:]: f1 = nn.functional.interpolate(f1, size=x.shape[2:])
        x = self.up3(torch.cat([x, f1], 1))
        
        return torch.tanh(self.final(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    def forward(self, img, sketch):
        return self.main(torch.cat([img, sketch], 1))

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print(f"🚀 Initializing SAFE Mode...")
    csv_path, root_dir = setup_dataset()
    if not csv_path: return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    # CLEANUP OLD IMAGES
    for f in glob.glob(f"{Config.OUTPUT_DIR}/*.png"): os.remove(f)

    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = SmartDataset(root_dir, csv_path, transform=tf)
    if len(dataset.photo_map) == 0: return

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    gen = SketchGenerator().to(Config.DEVICE)
    disc = Discriminator().to(Config.DEVICE)
    
    opt_g = optim.Adam(gen.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    print("💎 Training Started (Safe Mode)...")
    
    for epoch in range(Config.EPOCHS):
        fake = None
        for i, (photo, mask, target) in enumerate(dataloader):
            photo, mask, target = photo.to(Config.DEVICE), mask.to(Config.DEVICE), target.to(Config.DEVICE)
            
            # --- Train Generator ---
            opt_g.zero_grad()
            fake = gen(photo)
            
            # Safety Check: If image is flat (black/gray), penalize heavily
            if fake.std() < 0.01:
                # Force it to match target structure
                loss_g = l1(fake, target) * 1000 
            else:
                loss_gan = mse(disc(photo, fake), torch.ones_like(disc(photo, fake)))
                loss_pixel = (l1(fake, target) * mask).mean() * Config.LAMBDA_L1
                loss_g = loss_gan + loss_pixel

            loss_g.backward()
            
            # CLIP GRADIENTS (Prevents explosion)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt_g.step()
            
            # --- Train Discriminator ---
            opt_d.zero_grad()
            loss_d = (mse(disc(photo, target), torch.ones_like(disc(photo, target))) + 
                      mse(disc(photo, fake.detach()), torch.zeros_like(disc(photo, fake)))) * 0.5
            loss_d.backward()
            opt_d.step()
            
            if i % 10 == 0: print(f"[Epoch {epoch}][Batch {i}] Loss G: {loss_g.item():.4f}")
        
        if fake is not None:
            save_image(fake, f"{Config.OUTPUT_DIR}/epoch_{epoch}.png", normalize=True)
            print(f"📸 Saved epoch_{epoch}.png")

    torch.save(gen.state_dict(), "final_model.pth")
    print("✨ DONE.")

if __name__ == "__main__":
    train()