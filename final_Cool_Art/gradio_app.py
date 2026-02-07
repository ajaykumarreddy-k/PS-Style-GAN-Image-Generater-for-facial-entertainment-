"""
🎨 Production Hybrid Sketch Generator
Optimized CV2 + AI pipeline with ENHANCED_35 settings
"""

import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# MODEL ARCHITECTURE
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
    """VGG19-based encoder-decoder with skip connections"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc1 = nn.Sequential(*list(vgg.children())[:4])
        self.enc2 = nn.Sequential(*list(vgg.children())[4:14])
        self.enc3 = nn.Sequential(*list(vgg.children())[14:24])
        
        for p in self.parameters():
            p.requires_grad = False
        
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

# ==========================================
# LOAD MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "final_model_SHADING.pth"

try:
    netG = SketchGenerator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['model_state_dict'])
    else:
        netG.load_state_dict(checkpoint)
    netG.eval()
    print(f"✅ Model loaded successfully")
    print(f"🖥️  Device: {device}")
    model_loaded = True
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model_loaded = False

# ==========================================
# ENHANCED PROCESSING PIPELINE
# ==========================================
def create_enhanced_sketch(image, blur_kernel=18, ai_strength=35, show_intermediate=False):
    """
    ENHANCED_35 Pipeline - Production-optimized settings
    
    Args:
        image: PIL Image (RGB)
        blur_kernel: CV2 blur kernel (default: 18 for sharp lines)
        ai_strength: AI shading blend ratio 0-100 (default: 35 for optimal balance)
        show_intermediate: Return intermediate steps for debugging
    
    Returns:
        PIL Image or tuple of images if show_intermediate=True
    """
    if image is None:
        return None
    
    # STEP 1: CV2 Line Art with Enhanced Preprocessing
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Color dodge method
    inverted = 255 - gray
    k_size = (blur_kernel * 2) + 1
    blurred = cv2.GaussianBlur(inverted, (k_size, k_size), 0)
    sketch_cv2 = cv2.divide(gray, 255 - blurred, scale=256)
    
    # Sharpening filter for crisp edges
    kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                               [-0.5,  5.0, -0.5],
                               [-0.5, -0.5, -0.5]])
    sketch_cv2 = cv2.filter2D(sketch_cv2, -1, kernel_sharpen)
    sketch_cv2 = np.clip(sketch_cv2, 0, 255).astype(np.uint8)
    
    if not model_loaded:
        return Image.fromarray(sketch_cv2)
    
    # STEP 2: AI Shading Enhancement
    try:
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
        
        # Enhanced AI contrast
        generated = cv2.convertScaleAbs(generated, alpha=1.15, beta=-10)
        
        # STEP 3: Intelligent Blending
        alpha = ai_strength / 100.0
        final_sketch = cv2.addWeighted(generated, alpha, sketch_cv2, 1 - alpha, 0)
        
        # Final contrast boost
        final_sketch = cv2.convertScaleAbs(final_sketch, alpha=1.05, beta=0)
        
        if show_intermediate:
            return (Image.fromarray(sketch_cv2), 
                    Image.fromarray(generated), 
                    Image.fromarray(final_sketch))
        
        return Image.fromarray(final_sketch)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return Image.fromarray(sketch_cv2)

# ==========================================
# GRADIO INTERFACE
# ==========================================
def process_image(image, blur_k, strength):
    """Wrapper for Gradio interface"""
    return create_enhanced_sketch(image, blur_kernel=blur_k, ai_strength=strength)

# Custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

with gr.Blocks() as app:
    gr.HTML("""
        <div class="header">
            <h1>🎨 Hybrid Sketch Generator</h1>
            <p>Professional pencil sketches using CV2 + AI Enhanced Pipeline</p>
            <p><strong>Optimized ENHANCED_35 Configuration</strong></p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Input")
            input_image = gr.Image(type="pil", label="Upload Photo")
            
            gr.Markdown("### ⚙️ Advanced Settings")
            
            blur_slider = gr.Slider(
                minimum=10, 
                maximum=30, 
                value=18,
                step=1,
                label="🖊️ Line Sharpness",
                info="Lower = Sharper lines | Default: 18 (Recommended)"
            )
            
            strength_slider = gr.Slider(
                minimum=0, 
                maximum=100, 
                value=35,
                step=5,
                label="🎭 AI Shading Strength",
                info="0% = Pure CV2 | 100% = Pure AI | Default: 35% (Optimal)"
            )
            
            generate_btn = gr.Button("✨ Generate Sketch", variant="primary", size="lg")
            
            gr.Markdown("""
                ### 💡 Quick Tips
                - **Default settings** are optimized for best results
                - Works best with **portrait photos**
                - Try different AI strengths for artistic variation
                - 35% AI = Mathematical precision + Artistic depth
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### ✨ Output")
            output_image = gr.Image(type="pil", label="Enhanced Sketch")
            
            gr.Markdown("""
                ### 📊 Current Configuration
                - **Pipeline:** Hybrid CV2 + VGG19-GAN
                - **Model:** final_model_SHADING.pth (46MB)
                - **Optimization:** ENHANCED_35 preset
                - **Features:**
                  - ✅ Mathematical line precision
                  - ✅ AI shading enhancement
                  - ✅ Histogram equalization
                  - ✅ Edge sharpening
                  - ✅ Contrast optimization
            """)
    
    # Event handler
    generate_btn.click(
        fn=process_image,
        inputs=[input_image, blur_slider, strength_slider],
        outputs=output_image
    )
    
    # Examples section
    gr.Markdown("### 📌 Example")
    gr.Examples(
        examples=[["test_face.jpg"]],
        inputs=input_image,
        label="Try with sample image"
    )

# ==========================================
# LAUNCH
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 PRODUCTION HYBRID SKETCH GENERATOR")
    print("="*60)
    print(f"Status: {'✅ Model Loaded' if model_loaded else '⚠️  CV2 Fallback Mode'}")
    print(f"Device: {device}")
    print(f"Configuration: ENHANCED_35 (Optimized)")
    print("="*60 + "\n")
    
    app.launch(
        theme=gr.themes.Soft(),
        css=custom_css,
        share=False,
        server_name="127.0.0.1"
    )
