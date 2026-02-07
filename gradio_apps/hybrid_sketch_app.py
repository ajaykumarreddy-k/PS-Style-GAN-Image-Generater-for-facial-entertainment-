import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1️⃣ MODEL ARCHITECTURE (From stylesketch_shading.py)
# ==========================================
class SafeFusionModule(nn.Module):
    """Adds controlled noise for texture variation"""
    def __init__(self, channels):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + (noise * self.noise_scale)

class SketchGenerator(nn.Module):
    """
    VGG19-based encoder-decoder with skip connections.
    Trained to produce hyper-realistic graphite pencil sketches.
    """
    def __init__(self):
        super().__init__()
        # Encoder: Pre-trained VGG19 features (frozen)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc1 = nn.Sequential(*list(vgg.children())[:4])   # 64 channels
        self.enc2 = nn.Sequential(*list(vgg.children())[4:14]) # 256 channels
        self.enc3 = nn.Sequential(*list(vgg.children())[14:24])# 512 channels
        
        # Freeze encoder weights
        for p in self.parameters():
            p.requires_grad = False
        
        # Fusion module for texture
        self.fuse = SafeFusionModule(512)
        
        # Decoder: Transposed convolutions with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), 
            nn.InstanceNorm2d(256), 
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # 512 = 256 + 256 (skip)
            nn.InstanceNorm2d(128), 
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1),   # 192 = 128 + 64 (skip)
            nn.InstanceNorm2d(64), 
            nn.ReLU(True)
        )
        self.final = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        # Encode
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        # Decode with skip connections
        x = self.up1(self.fuse(f3))
        
        # Resize f2 to match x if needed
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:]) if f2.shape[2:] != x.shape[2:] else f2
        x = self.up2(torch.cat([x, f2_rs], 1))
        
        # Resize f1 to match x if needed
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:]) if f1.shape[2:] != x.shape[2:] else f1
        x = self.up3(torch.cat([x, f1_rs], 1))
        
        # Output: Single channel sketch (tanh range: -1 to 1)
        out = torch.tanh(self.final(x))
        return out

# ==========================================
# 2️⃣ LOAD THE .PTH MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/prince/ProjectsMain/Project_Vibe/final_model_SHADING.pth"

try:
    # Initialize model architecture
    netG = SketchGenerator().to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both full checkpoint dicts and raw state_dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['model_state_dict'])
    else:
        netG.load_state_dict(checkpoint)
    
    netG.eval()
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"🖥️  Running on: {device}")
    model_loaded = True
    
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("ℹ️  Will fallback to pure CV2 mode")
    model_loaded = False

# ==========================================
# 3️⃣ IMAGE PROCESSING PIPELINE
# ==========================================
def process_pipeline(image, blur_k, strength):
    """
    Hybrid pipeline: CV2 (Structure) + GAN (Shading)
    
    Args:
        image: PIL Image (RGB)
        blur_k: CV2 blur kernel size (1-50)
        strength: AI shading blend ratio (0-100%)
    
    Returns:
        PIL Image: Final sketch result
    """
    if image is None:
        return None
    
    # --- STEP 1: CLEAN LINE ART (CV2 Color Dodge Method) ---
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Invert & Blur (The "Color Dodge" Method from Photoshop)
    inverted = 255 - gray
    k_size = (blur_k * 2) + 1  # Ensure odd kernel size
    blurred = cv2.GaussianBlur(inverted, (k_size, k_size), 0)
    
    # Create Base Sketch: gray / (255 - blurred) * 256
    # This creates the "pencil on paper" effect
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    # --- STEP 2: APPLY AI SHADING (.pth GAN Model) ---
    if not model_loaded:
        # Fallback: Return pure CV2 sketch if model failed to load
        return Image.fromarray(sketch)
    
    try:
        # Prepare sketch for AI (need RGB input for VGG19)
        # The model expects 3-channel RGB, but we'll convert sketch to grayscale after
        sketch_pil = Image.fromarray(sketch)
        
        # Convert grayscale sketch to RGB (replicate channel 3 times)
        sketch_rgb = sketch_pil.convert('RGB')
        
        # Normalize to [-1, 1] range (model's expected input)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Model trained on 256x256
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(sketch_rgb).unsqueeze(0).to(device)
        
        # Run Inference (Model predicts enhanced sketch with shading)
        with torch.no_grad():
            generated = netG(input_tensor)
            
            # Post-process: Convert from [-1, 1] back to [0, 255]
            generated = generated.squeeze().cpu().detach().numpy()
            generated = (generated * 0.5 + 0.5) * 255.0
            generated = np.clip(generated, 0, 255).astype(np.uint8)
            
            # Resize back to original image size if needed
            if generated.shape != sketch.shape:
                generated = cv2.resize(generated, (sketch.shape[1], sketch.shape[0]), 
                                     interpolation=cv2.INTER_LANCZOS4)
            
            # --- STEP 3: BLEND CV2 + AI (Strength Control) ---
            # strength slider (0-100) controls opacity
            # 0% = Pure CV2 (mathematical precision)
            # 100% = Pure AI (artistic shading)
            alpha = strength / 100.0
            final_output = cv2.addWeighted(generated, alpha, sketch, 1 - alpha, 0)
            
            return Image.fromarray(final_output)
            
    except Exception as e:
        print(f"Inference Error: {e}")
        # Fallback to pure CV2 if anything fails
        return Image.fromarray(sketch)

# ==========================================
# 4️⃣ GRADIO UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Hybrid Sketch: CV2 Lines + GAN Shading
        
        **Pipeline:** Photo → CV2 Line Art → AI Enhancement → Hyper-realistic Graphite Sketch
        
        > "Hyper-realistic graphite pencil sketch, intricate charcoal shading, smooth volumetric lighting, 
        > rough paper texture, high contrast, architectural precision, masterful cross-hatching, 8k resolution, 
        > solid strokes, no smudge."
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="📸 Input Photo")
            
            with gr.Row():
                blur_slider = gr.Slider(
                    minimum=1, 
                    maximum=50, 
                    value=21, 
                    step=1,
                    label="🖊️ CV2 Line Softness",
                    info="Lower = Sharp lines, Higher = Soft/blurred lines"
                )
                
            with gr.Row():
                strength_slider = gr.Slider(
                    minimum=0, 
                    maximum=100, 
                    value=75, 
                    step=1,
                    label="🎭 AI Shading Strength",
                    info="0% = Pure CV2, 100% = Pure AI Model"
                )
            
            run_btn = gr.Button("🚀 Generate Sketch", variant="primary", size="lg")
        
        with gr.Column():
            output_img = gr.Image(type="pil", label="✨ AI Shaded Sketch")
    
    # Event handler
    run_btn.click(
        fn=process_pipeline, 
        inputs=[input_img, blur_slider, strength_slider], 
        outputs=output_img
    )
    
    # Example section
    gr.Markdown("### 📌 Tips")
    gr.Markdown(
        """
        - **CV2 Line Softness (21)**: Start with default, adjust if lines are too sharp/soft
        - **AI Shading Strength (75%)**: Balanced blend of precision + artistry
        - Upload portrait photos for best results (the model was trained on faces)
        """
    )

# ==========================================
# 5️⃣ LAUNCH
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 HYBRID SKETCH GENERATOR")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if model_loaded else '⚠️  Fallback to CV2'}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    demo.launch(share=False)
