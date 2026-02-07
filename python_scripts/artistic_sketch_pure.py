import cv2
import numpy as np
from PIL import Image

def artistic_sketch_converter(image_path, output_path):
    """
    Pure CV2 Artistic Sketch Converter
    Creates hyper-realistic graphite pencil sketches with:
    - Clean line art
    - Volumetric shading
    - Cross-hatching simulation
    - Paper texture
    - Charcoal effects
    """
    
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ==========================================
    # LAYER 1: CLEAN LINE ART (Color Dodge)
    # ==========================================
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch_base = cv2.divide(gray, 255 - blurred, scale=256)
    
    # ==========================================
    # LAYER 2: VOLUMETRIC SHADING (Bilateral Filter)
    # ==========================================
    # Bilateral filter preserves edges while smoothing
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Create shadow map (darker areas)
    shadow_map = cv2.subtract(255, smooth)
    shadow_map = cv2.GaussianBlur(shadow_map, (15, 15), 0)
    
    # Enhance shadows using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_shadows = clahe.apply(shadow_map)
    
    # Invert back to get shading layer
    shading_layer = 255 - enhanced_shadows
    
    # ==========================================
    # LAYER 3: DETAIL ENHANCEMENT (Unsharp Mask)
    # ==========================================
    gaussian = cv2.GaussianBlur(sketch_base, (0, 0), 2.0)
    unsharp = cv2.addWeighted(sketch_base, 1.5, gaussian, -0.5, 0)
    
    # ==========================================
    # LAYER 4: CROSS-HATCHING SIMULATION
    # ==========================================
    # Create directional gradients for hatching effect
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
    
    # Invert to get hatching lines
    hatching = 255 - gradient_mag
    hatching = cv2.GaussianBlur(hatching, (3, 3), 0)
    
    # ==========================================
    # LAYER 5: PAPER TEXTURE
    # ==========================================
    # Create realistic paper grain
    h, w = gray.shape
    noise = np.random.normal(128, 15, (h, w)).astype(np.uint8)
    
    # Apply texture only to non-white areas
    texture_mask = cv2.threshold(sketch_base, 250, 255, cv2.THRESH_BINARY_INV)[1]
    paper_texture = cv2.bitwise_and(noise, noise, mask=texture_mask)
    
    # ==========================================
    # LAYER 6: COMBINE ALL LAYERS
    # ==========================================
    # Start with clean lines (60% weight)
    result = cv2.addWeighted(unsharp, 0.6, shading_layer, 0.4, 0)
    
    # Add hatching for depth (subtle)
    result = cv2.addWeighted(result, 0.85, hatching, 0.15, 0)
    
    # Add paper texture
    result = cv2.subtract(result, paper_texture // 8)
    
    # ==========================================
    # FINAL POLISH: CONTRAST & BRIGHTNESS
    # ==========================================
    # Increase contrast for "graphite pop"
    alpha = 1.3  # Contrast
    beta = -20   # Brightness (darker for graphite feel)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # Crush the blacks (make dark areas darker)
    result = cv2.threshold(result, 245, 255, cv2.THRESH_TRUNC)[1]
    
    # Final normalization
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    # ==========================================
    # OPTIONAL: CHARCOAL EFFECT (Heavy Mode)
    # ==========================================
    # Uncomment for even more dramatic artistic effect
    # kernel = np.ones((2, 2), np.uint8)
    # result = cv2.erode(result, kernel, iterations=1)
    
    # Save result
    cv2.imwrite(output_path, result)
    return result

def artistic_sketch_converter_premium(image_path, output_path):
    """
    PREMIUM VERSION: Maximum artistic quality
    Uses advanced techniques for professional-grade sketches
    """
    
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast first
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # ==========================================
    # TECHNIQUE 1: Multi-Scale Color Dodge
    # ==========================================
    sketches = []
    blur_sizes = [11, 21, 31]  # Multiple blur kernels
    
    for size in blur_sizes:
        inverted = 255 - enhanced
        blurred = cv2.GaussianBlur(inverted, (size, size), 0)
        sketch = cv2.divide(enhanced, 255 - blurred, scale=256)
        sketches.append(sketch)
    
    # Combine multi-scale sketches
    base_sketch = cv2.addWeighted(sketches[0], 0.3, sketches[1], 0.5, 0)
    base_sketch = cv2.addWeighted(base_sketch, 1.0, sketches[2], 0.2, 0)
    
    # ==========================================
    # TECHNIQUE 2: Edge-Preserving Detail
    # ==========================================
    # Use edge detection for fine details
    edges = cv2.Canny(enhanced, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_inv = 255 - edges_dilated
    
    # Combine with base sketch
    detailed_sketch = cv2.bitwise_and(base_sketch, edges_inv)
    
    # ==========================================
    # TECHNIQUE 3: Artistic Shading
    # ==========================================
    # Create soft shading using morphological operations
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shading = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel_large)
    shading = cv2.GaussianBlur(shading, (15, 15), 0)
    shading_inv = 255 - shading
    
    # ==========================================
    # TECHNIQUE 4: Pencil Stroke Simulation
    # ==========================================
    # Create directional blur for stroke effect
    kernel_motion = np.zeros((9, 9))
    kernel_motion[4, :] = np.ones(9)
    kernel_motion = kernel_motion / 9
    
    strokes = cv2.filter2D(detailed_sketch, -1, kernel_motion)
    
    # ==========================================
    # COMBINE ALL LAYERS
    # ==========================================
    # Layer 1: Base sketch (50%)
    result = cv2.addWeighted(detailed_sketch, 0.5, strokes, 0.3, 0)
    
    # Layer 2: Add shading (20%)
    result = cv2.addWeighted(result, 1.0, shading_inv, 0.2, 0)
    
    # ==========================================
    # FINAL ARTISTIC TOUCHES
    # ==========================================
    # 1. Add subtle paper grain
    h, w = result.shape
    grain = np.random.normal(0, 8, (h, w)).astype(np.int16)
    result = np.clip(result.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    
    # 2. Enhance contrast
    result = cv2.convertScaleAbs(result, alpha=1.2, beta=-10)
    
    # 3. Vignette effect (darker edges like real paper)
    rows, cols = result.shape
    kernel_x = cv2.getGaussianKernel(cols, cols / 3)
    kernel_y = cv2.getGaussianKernel(rows, rows / 3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    result = (result * mask).astype(np.uint8)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save result
    cv2.imwrite(output_path, result)
    return result


if __name__ == "__main__":
    print("🎨 PURE ARTISTIC SKETCH CONVERTER")
    print("=" * 60)
    
    # Standard version
    print("\n1️⃣ Creating STANDARD artistic sketch...")
    result1 = artistic_sketch_converter(
        "test_face.jpg", 
        "artistic_result_STANDARD.png"
    )
    print("✅ Saved: artistic_result_STANDARD.png")
    
    # Premium version
    print("\n2️⃣ Creating PREMIUM artistic sketch...")
    result2 = artistic_sketch_converter_premium(
        "test_face.jpg",
        "artistic_result_PREMIUM.png"
    )
    print("✅ Saved: artistic_result_PREMIUM.png")
    
    print("\n" + "=" * 60)
    print("🎨 Done! Check the outputs:")
    print("   - artistic_result_STANDARD.png (balanced)")
    print("   - artistic_result_PREMIUM.png (maximum artistry)")
