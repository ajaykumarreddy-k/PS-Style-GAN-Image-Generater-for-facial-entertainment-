import gradio as gr
import cv2
import numpy as np
from PIL import Image

def perfect_artistic_sketch(image):
    """
    PERFECT ARTISTIC SKETCH - Pure Art Quality
    No AI model needed - pure CV2 mastery
    """
    if image is None:
        return None
    
    # Convert PIL to OpenCV
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # ==========================================
    # STEP 1: ENHANCED BASE LINE ART
    # ==========================================
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    inverted = 255 - enhanced
    
    # Multi-scale color dodge
    blur1 = cv2.GaussianBlur(inverted, (25, 25), 0)
    sketch_soft = cv2.divide(enhanced, 255 - blur1, scale=256)
    
    blur2 = cv2.GaussianBlur(inverted, (11, 11), 0)
    sketch_detail = cv2.divide(enhanced, 255 - blur2, scale=256)
    
    base_lines = cv2.addWeighted(sketch_soft, 0.7, sketch_detail, 0.3, 0)
    
    # ==========================================
    # STEP 2: GENTLE VOLUMETRIC SHADING
    # ==========================================
    smooth = cv2.bilateralFilter(gray, 9, 100, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    shadows = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel)
    shadows = cv2.GaussianBlur(shadows, (25, 25), 0)
    shadows = cv2.normalize(shadows, None, 0, 50, cv2.NORM_MINMAX)
    
    # ==========================================
    # STEP 3: DELICATE TEXTURE
    # ==========================================
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)
    laplacian = np.uint8(laplacian / laplacian.max() * 255)
    laplacian_inv = 255 - laplacian
    texture = cv2.GaussianBlur(laplacian_inv, (3, 3), 0)
    
    # ==========================================
    # STEP 4: ARTISTIC PAPER GRAIN
    # ==========================================
    h, w = gray.shape
    grain = np.random.normal(0, 3, (h, w)).astype(np.int16)
    
    # ==========================================
    # STEP 5: INTELLIGENT COMBINATION
    # ==========================================
    result = base_lines.copy().astype(np.float32)
    result = result - shadows
    result = cv2.addWeighted(result.astype(np.uint8), 0.92, texture, 0.08, 0)
    result = np.clip(result.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    
    # ==========================================
    # STEP 6: FINAL ARTISTIC TOUCH
    # ==========================================
    result = cv2.convertScaleAbs(result, alpha=1.15, beta=-5)
    result = cv2.threshold(result, 248, 255, cv2.THRESH_TRUNC)[1]
    result = cv2.normalize(result, None, 5, 255, cv2.NORM_MINMAX)
    
    return Image.fromarray(result)


def adjustable_artistic_sketch(image, shading_strength, detail_level, contrast):
    """
    Adjustable version with user controls
    """
    if image is None:
        return None
    
    # Convert PIL to OpenCV
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Enhanced base
    clahe_clip = 2.0 + (detail_level / 100.0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    inverted = 255 - enhanced
    
    # Adaptive blur based on detail level
    blur_size = int(25 - (detail_level / 10))
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = max(11, min(31, blur_size))
    
    blur1 = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
    sketch_soft = cv2.divide(enhanced, 255 - blur1, scale=256)
    
    blur2 = cv2.GaussianBlur(inverted, (11, 11), 0)
    sketch_detail = cv2.divide(enhanced, 255 - blur2, scale=256)
    
    base_lines = cv2.addWeighted(sketch_soft, 0.7, sketch_detail, 0.3, 0)
    
    # Shading (controlled by slider)
    smooth = cv2.bilateralFilter(gray, 9, 100, 100)
    kernel_size = int(15 + (shading_strength / 10))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    shadows = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel)
    shadows = cv2.GaussianBlur(shadows, (25, 25), 0)
    
    shadow_max = int(30 + (shading_strength / 2))
    shadows = cv2.normalize(shadows, None, 0, shadow_max, cv2.NORM_MINMAX)
    
    # Texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)
    laplacian = np.uint8(laplacian / laplacian.max() * 255)
    laplacian_inv = 255 - laplacian
    texture = cv2.GaussianBlur(laplacian_inv, (3, 3), 0)
    
    # Paper grain
    h, w = gray.shape
    grain_strength = 2 + (detail_level / 50)
    grain = np.random.normal(0, grain_strength, (h, w)).astype(np.int16)
    
    # Combine
    result = base_lines.copy().astype(np.float32)
    result = result - shadows
    result = cv2.addWeighted(result.astype(np.uint8), 0.92, texture, 0.08, 0)
    result = np.clip(result.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    
    # Final adjustments with contrast control
    alpha = 1.0 + (contrast / 100.0)
    beta = -5 - (contrast / 10)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    result = cv2.threshold(result, 248, 255, cv2.THRESH_TRUNC)[1]
    result = cv2.normalize(result, None, 5, 255, cv2.NORM_MINMAX)
    
    return Image.fromarray(result)


# ==========================================
# GRADIO UI
# ==========================================
with gr.Blocks(title="Pure Artistic Sketch Converter") as demo:
    gr.Markdown(
        """
        # 🎨 Pure Artistic Sketch Converter
        
        **Professional-grade pencil sketches using pure CV2 artistry**
        
        No AI models. No neural networks. Just pure computer vision mastery.
        
        > *"Hyper-realistic graphite pencil sketch, intricate charcoal shading, 
        > smooth volumetric lighting, rough paper texture, high contrast, 
        > architectural precision, masterful cross-hatching, 8k resolution, 
        > solid strokes, no smudge."*
        """
    )
    
    with gr.Tabs():
        # Tab 1: One-Click Perfect
        with gr.Tab("✨ One-Click Perfect"):
            gr.Markdown("### Instant artistic perfection - just upload and click!")
            
            with gr.Row():
                with gr.Column():
                    input_simple = gr.Image(type="pil", label="📸 Upload Photo")
                    btn_simple = gr.Button("🎨 Create Perfect Sketch", variant="primary", size="lg")
                
                with gr.Column():
                    output_simple = gr.Image(type="pil", label="✨ Artistic Sketch")
            
            btn_simple.click(
                fn=perfect_artistic_sketch,
                inputs=input_simple,
                outputs=output_simple
            )
        
        # Tab 2: Advanced Controls
        with gr.Tab("🎛️ Advanced Controls"):
            gr.Markdown("### Fine-tune your artistic style")
            
            with gr.Row():
                with gr.Column():
                    input_advanced = gr.Image(type="pil", label="📸 Upload Photo")
                    
                    shading_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="🌗 Shading Strength",
                        info="0 = Light sketch, 100 = Deep shadows"
                    )
                    
                    detail_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="🔍 Detail Level",
                        info="0 = Soft/artistic, 100 = Sharp/precise"
                    )
                    
                    contrast_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=15,
                        step=1,
                        label="⚡ Contrast Boost",
                        info="0 = Subtle, 50 = Dramatic"
                    )
                    
                    btn_advanced = gr.Button("🎨 Generate Custom Sketch", variant="primary", size="lg")
                
                with gr.Column():
                    output_advanced = gr.Image(type="pil", label="✨ Custom Artistic Sketch")
            
            btn_advanced.click(
                fn=adjustable_artistic_sketch,
                inputs=[input_advanced, shading_slider, detail_slider, contrast_slider],
                outputs=output_advanced
            )
    
    gr.Markdown(
        """
        ---
        ### 📌 Tips for Best Results
        
        - **Portrait photos** work exceptionally well
        - **Good lighting** in original photo → better sketch details
        - **High resolution** images produce cleaner line work
        
        ### 🎨 What Makes This Art?
        
        ✨ **Multi-scale color dodge** - Clean, precise line extraction  
        ✨ **Volumetric shading** - Depth and dimension like real graphite  
        ✨ **Delicate texture** - Hair strands and fine details  
        ✨ **Paper grain** - Authentic sketch-on-paper feel  
        ✨ **Intelligent blending** - Professional artist-level composition  
        """
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 PURE ARTISTIC SKETCH CONVERTER")
    print("="*60)
    print("Professional-grade pencil sketches")
    print("No AI models required - Pure CV2 artistry")
    print("="*60 + "\n")
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
