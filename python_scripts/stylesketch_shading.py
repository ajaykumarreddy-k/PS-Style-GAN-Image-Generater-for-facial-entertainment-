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
# 1. CONFIG (Risk Mode: Shading)
# ==========================================
class Config:
    IMG_SIZE = 256
    BATCH_SIZE = 2      
    EPOCHS = 30          # Short run! We just want to add texture.
    LR = 0.00005         # LOW LR to protect the structure.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ZIP_NAME = "SKSF-A.zip"
    EXTRACT_DIR = "./dataset_extracted"
    
    # NEW OUTPUTS
    OUTPUT_DIR = "./output_shading"  # New folder for safety
    LOAD_MODEL = "final_model_CHAMPION.pth" # Start from the winner
    SAVE_MODEL = "final_model_SHADING.pth"
    
    # --- THE SHADING RECIPE ---
    LAMBDA_L1 = 10       
    LAMBDA_STYLE = 2000  # High texture weight
    LAMBDA_PERC = 10     
    LAMBDA_GAN = 1      
    LAMBDA_TV = 0.5      
    LAMBDA_GRAY = 150    # THE RISK: Forces the model to draw gray mass (shadows)

# ==========================================
# 2. DATASET
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
    a, b, c, d = input.size() 
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def stroke_continuity_loss(img):
    h_diff = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    v_diff = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(h_diff) + torch.mean(v_diff)

def gray_mass_loss(img, target):
    """
    Forces the model to match the overall darkness (graphite density) of the target.
    """
    img_pool = torch.nn.functional.avg_pool2d(img, 16)
    target_pool = torch.nn.functional.avg_pool2d(target, 16)
    return torch.mean(torch.abs(img_pool - target_pool))

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    print(f"🚀 Initializing SHADING (RISK) Mode...")
    csv_path, root_dir = setup_dataset()
    if not csv_path: return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = SmartDataset(root_dir, csv_path, transform=tf)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    gen = SketchGenerator().to(Config.DEVICE)
    disc = Discriminator().to(Config.DEVICE)
    
    # --- VITAL: START FROM CHAMPION ---
    if os.path.exists(Config.LOAD_MODEL):
        try:
            gen.load_state_dict(torch.load(Config.LOAD_MODEL))
            print(f"✅ Loaded CHAMPION model: {Config.LOAD_MODEL}")
            print("🎨 Applying Graphite Shading layers now...")
        except Exception as e:
            print(f"❌ Error loading Champion: {e}")
            return
    else:
        print(f"❌ STOP! {Config.LOAD_MODEL} is missing. Do not train from scratch!")
        return

    # Lower LR for fine-tuning
    opt_g = optim.Adam(gen.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    for epoch in range(Config.EPOCHS):
        fake = None
        for i, (photo, mask, target) in enumerate(dataloader):
            photo, mask, target = photo.to(Config.DEVICE), mask.to(Config.DEVICE), target.to(Config.DEVICE)
            opt_g.zero_grad()
            
            # Augmentation (Keep it, helps stability)
            if random.random() < 0.5:
                h, w = photo.shape[2], photo.shape[3]
                y, x = random.randint(0, h-60), random.randint(0, w-60)
                photo_aug = photo.clone()
                photo_aug[:, :, y:y+60, x:x+60] = 0
            else: photo_aug = photo

            # 1. Generate
            fake, fake_feats = gen(photo_aug, return_features=True)
            target_3ch = target.repeat(1, 3, 1, 1)
            with torch.no_grad(): real_feats = gen.extract_features(target_3ch)
            
            # 2. Losses
            loss_gan = mse(disc(photo, fake), torch.ones_like(disc(photo, fake)))
            loss_pixel = (l1(fake, target) * mask).mean() * Config.LAMBDA_L1
            
            loss_perc = 0
            for f_fake, f_real in zip(fake_feats, real_feats):
                loss_perc += l1(f_fake, f_real.detach())
            loss_perc *= Config.LAMBDA_PERC
            
            loss_style = 0
            for f_fake, f_real in zip(fake_feats, real_feats):
                loss_style += l1(gram_matrix(f_fake), gram_matrix(f_real.detach()))
            loss_style *= Config.LAMBDA_STYLE

            loss_tv = stroke_continuity_loss(fake) * Config.LAMBDA_TV
            
            # NEW: GRAY MASS LOSS (The Shading Force)
            loss_gray = gray_mass_loss(fake, target) * Config.LAMBDA_GRAY

            # Total
            loss_g = (loss_gan * Config.LAMBDA_GAN) + loss_pixel + loss_perc + loss_style + loss_tv + loss_gray
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt_g.step()
            
            # --- Disc ---
            opt_d.zero_grad()
            loss_d = (mse(disc(photo, target), torch.ones_like(disc(photo, target))) + 
                      mse(disc(photo, fake.detach()), torch.zeros_like(disc(photo, fake)))) * 0.5
            loss_d.backward()
            opt_d.step()
            
            if i % 10 == 0: 
                print(f"[Ep {epoch}][Bt {i}] GrayMass:{loss_gray:.2f} Style:{loss_style:.1f}")
        
        if fake is not None:
            save_image(fake, f"{Config.OUTPUT_DIR}/epoch_{epoch}.png", normalize=True)
            print(f"📸 Saved epoch_{epoch}.png")

    torch.save(gen.state_dict(), Config.SAVE_MODEL)
    print(f"✨ DONE. Saved as {Config.SAVE_MODEL}")

if __name__ == "__main__":
    train()