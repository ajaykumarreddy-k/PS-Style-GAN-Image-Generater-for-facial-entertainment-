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
# 1. CONFIG (Final Artistic Mode v2)
# ==========================================
class Config:
    IMG_SIZE = 256
    BATCH_SIZE = 2      
    EPOCHS = 80          
    LR = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ZIP_NAME = "SKSF-A.zip"
    EXTRACT_DIR = "./dataset_extracted"
    OUTPUT_DIR = "./output_artistic"
    
    # NEW MODEL NAME
    MODEL_NAME = "final_model_artistic1.pth"
    
    # --- THE ARTISTIC RECIPE ---
    LAMBDA_L1 = 10       # Structure (Low to allow artistic freedom)
    LAMBDA_STYLE = 1000  # Texture (High to force cross-hatching)
    LAMBDA_PERC = 10     # Feature Identity
    LAMBDA_GAN = 1       # Realism
    LAMBDA_TV = 0.5      # NEW: Stroke Continuity (Smooths out jitter)

# ==========================================
# 2. DATASET (Robust)
# ==========================================
def setup_dataset():
    if not os.path.exists(Config.EXTRACT_DIR):
        if os.path.exists(Config.ZIP_NAME):
            with zipfile.ZipFile(Config.ZIP_NAME, 'r') as z:
                z.extractall(Config.EXTRACT_DIR)

    possible_roots = glob.glob(f"{Config.EXTRACT_DIR}/**/Photo", recursive=True)
    if not possible_roots: possible_roots = glob.glob(f"{Config.EXTRACT_DIR}/**/photo", recursive=True)
    if not possible_roots: return None, None
    root_data_dir = os.path.dirname(possible_roots[0])
    csv_files = glob.glob(f"{root_data_dir}/*.csv")
    if not csv_files: csv_files = glob.glob(f"{Config.EXTRACT_DIR}/**/*.csv", recursive=True)
    if not csv_files: return None, None
    return csv_files[0], root_data_dir

class SmartDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        try: self.df = pd.read_csv(csv_file)
        except: self.df = pd.DataFrame()
        self.photo_map = self._scan(root_dir, "Photo")
        self.mask_map = self._scan(root_dir, "mask")
        self.style_maps = {i: self._scan(root_dir, f"style{i}") for i in range(1, 8)}
        self.id_col = 0
        for col in range(len(self.df.columns)):
            if str(self.df.iloc[0, col]).strip() in self.photo_map:
                self.id_col = col; break

    def _scan(self, root, folder):
        target = os.path.join(root, folder)
        if not os.path.exists(target):
            target = os.path.join(root, folder.lower())
            if not os.path.exists(target): return {}
        mapping = {}
        for f in os.listdir(target):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                mapping[os.path.splitext(f)[0]] = f
        return mapping

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0: return None
        fid = str(self.df.iloc[idx, self.id_col]).strip()
        if fid not in self.photo_map: return self.__getitem__((idx+1)%len(self.df))
        
        p_path = os.path.join(self.root_dir, "Photo", self.photo_map[fid])
        m_name = self.mask_map.get(fid)
        m_path = os.path.join(self.root_dir, "mask", m_name) if m_name else None
        
        s_idx = random.randint(1,7)
        s_map = self.style_maps.get(s_idx, {})
        s_name = s_map.get(fid)
        if not s_name: 
             for i in range(1,8):
                 if fid in self.style_maps.get(i, {}):
                     s_name = self.style_maps[i][fid]; s_idx = i; break
        s_path = os.path.join(self.root_dir, f"style{s_idx}", s_name) if s_name else None
        if not s_path: return self.__getitem__((idx+1)%len(self.df))

        try:
            photo = Image.open(p_path).convert("RGB")
            sketch = Image.open(s_path).convert("L")
            if m_path: mask = Image.open(m_path).convert("L")
            else: mask = Image.new("L", photo.size, 255)

            if self.transform:
                photo = self.transform(photo)
                mask = self.transform(mask)
                sketch = self.transform(sketch)
            return photo, mask, sketch
        except: return self.__getitem__((idx+1)%len(self.df))

# ==========================================
# 3. ARCHITECTURE
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
        for p in self.parameters(): p.requires_grad = False
        
        self.fuse = SafeFusionModule(512)
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.final = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x, return_features=False):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        x = self.up1(self.fuse(f3))
        
        # Skip Connections with Resize Safety
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:]) if f2.shape[2:]!=x.shape[2:] else f2
        x = self.up2(torch.cat([x, f2_rs], 1))
        
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:]) if f1.shape[2:]!=x.shape[2:] else f1
        x = self.up3(torch.cat([x, f1_rs], 1))
        
        out = torch.tanh(self.final(x))
        if return_features: return out, [f1, f2, f3]
        return out
        
    def extract_features(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return [f1, f2, f3]

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
# 4. LOSS FUNCTIONS
# ==========================================
def gram_matrix(input):
    """Calculates texture correlation (Style)"""
    a, b, c, d = input.size() 
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def stroke_continuity_loss(img):
    """Forces lines to be smooth continuous strokes (Total Variation)"""
    h_diff = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    v_diff = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(h_diff) + torch.mean(v_diff)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    print(f"🚀 Initializing FINAL ARTISTIC Mode (v1)...")
    csv_path, root_dir = setup_dataset()
    if not csv_path: return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = SmartDataset(root_dir, csv_path, transform=tf)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    gen = SketchGenerator().to(Config.DEVICE)
    disc = Discriminator().to(Config.DEVICE)
    
    # Check for existing checkpoint of THIS version
    if os.path.exists(Config.MODEL_NAME):
        try:
            gen.load_state_dict(torch.load(Config.MODEL_NAME))
            print(f"🔄 Resuming from {Config.MODEL_NAME}...")
        except: pass

    opt_g = optim.Adam(gen.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    print("💎 Training Started (Texture + Cutout Augmentation)...")
    
    for epoch in range(Config.EPOCHS):
        fake = None
        for i, (photo, mask, target) in enumerate(dataloader):
            photo, mask, target = photo.to(Config.DEVICE), mask.to(Config.DEVICE), target.to(Config.DEVICE)
            
            # --- Train Generator ---
            opt_g.zero_grad()
            
            # NEW: CUTOUT AUGMENTATION (Force Imagination)
            if random.random() < 0.5: # 50% chance
                h, w = photo.shape[2], photo.shape[3]
                y = random.randint(0, h - 60)
                x = random.randint(0, w - 60)
                photo_aug = photo.clone()
                photo_aug[:, :, y:y+60, x:x+60] = 0 # Erase a box
            else:
                photo_aug = photo
            
            # 1. Generate & Extract
            fake, fake_feats = gen(photo_aug, return_features=True)
            
            target_3ch = target.repeat(1, 3, 1, 1)
            with torch.no_grad(): real_feats = gen.extract_features(target_3ch)
            
            # 2. Basic Losses
            loss_gan = mse(disc(photo, fake), torch.ones_like(disc(photo, fake)))
            loss_pixel = (l1(fake, target) * mask).mean() * Config.LAMBDA_L1
            
            # 3. Perceptual Loss
            loss_perc = 0
            for f_fake, f_real in zip(fake_feats, real_feats):
                loss_perc += l1(f_fake, f_real.detach())
            loss_perc = loss_perc * Config.LAMBDA_PERC
            
            # 4. Style Loss (Texture)
            loss_style = 0
            for f_fake, f_real in zip(fake_feats, real_feats):
                G_fake = gram_matrix(f_fake)
                G_real = gram_matrix(f_real.detach())
                loss_style += l1(G_fake, G_real)
            loss_style = loss_style * Config.LAMBDA_STYLE

            # 5. NEW: Stroke Continuity Loss
            loss_tv = stroke_continuity_loss(fake) * Config.LAMBDA_TV

            # Total G Loss
            loss_g = (loss_gan * Config.LAMBDA_GAN) + loss_pixel + loss_perc + loss_style + loss_tv
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt_g.step()
            
            # --- Train Discriminator ---
            opt_d.zero_grad()
            loss_d = (mse(disc(photo, target), torch.ones_like(disc(photo, target))) + 
                      mse(disc(photo, fake.detach()), torch.zeros_like(disc(photo, fake)))) * 0.5
            loss_d.backward()
            opt_d.step()
            
            if i % 10 == 0: 
                print(f"[Ep {epoch}][Bt {i}] Px:{loss_pixel:.1f} Sty:{loss_style:.1f} TV:{loss_tv:.2f}")
        
        if fake is not None:
            save_image(fake, f"{Config.OUTPUT_DIR}/epoch_{epoch}.png", normalize=True)
            print(f"📸 Saved epoch_{epoch}.png")

    # SAVE WITH NEW NAME
    torch.save(gen.state_dict(), Config.MODEL_NAME)
    print(f"✨ DONE. Model saved as {Config.MODEL_NAME}")

if __name__ == "__main__":
    train()