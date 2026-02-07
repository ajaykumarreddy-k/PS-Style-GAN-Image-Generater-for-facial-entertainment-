import cv2
import numpy as np

def perfect_artistic_sketch(image_path, output_path):
    """
    PERFECT ARTISTIC SKETCH
    The ultimate balance of clean lines, soft shading, and artistic detail
    """
    
    # Load and prepare
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ==========================================
    # STEP 1: ENHANCED BASE LINE ART
    # ==========================================
    # Use CLAHE for better detail
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Multi-scale color dodge (lighttouch + fine detail)
    inverted = 255 - enhanced
    
    # Light blur for main lines
    blur1 = cv2.GaussianBlur(inverted, (25, 25), 0)
    sketch_soft = cv2.divide(enhanced, 255 - blur1, scale=256)
    
    # Fine blur for details
    blur2 = cv2.GaussianBlur(inverted, (11, 11), 0)
    sketch_detail = cv2.divide(enhanced, 255 - blur2, scale=256)
    
    # Combine: 70% soft + 30% detail
    base_lines = cv2.addWeighted(sketch_soft, 0.7, sketch_detail, 0.3, 0)
    
    # ==========================================
    # STEP 2: GENTLE VOLUMETRIC SHADING
    # ==========================================
    # Bilateral filter for smooth shading
    smooth = cv2.bilateralFilter(gray, 9, 100, 100)
    
    # Create subtle shadow map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    shadows = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel)
    shadows = cv2.GaussianBlur(shadows, (25, 25), 0)
    
    # Normalize shadows (very subtle)
    shadows = cv2.normalize(shadows, None, 0, 50, cv2.NORM_MINMAX)  # Max 50 darkness
    
    # ==========================================
    # STEP 3: DELICATE HAIR TEXTURE
    # ==========================================
    # Detect fine details (hair strands, etc.)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)
    laplacian = np.uint8(laplacian / laplacian.max() * 255)
    laplacian_inv = 255 - laplacian
    
    # Smooth for artistic effect
    texture = cv2.GaussianBlur(laplacian_inv, (3, 3), 0)
    
    # ==========================================
    # STEP 4: ARTISTIC PAPER GRAIN (SUBTLE)
    # ==========================================
    h, w = gray.shape
    # Very subtle noise
    grain = np.random.normal(0, 3, (h, w)).astype(np.int16)
    
    # ==========================================
    # STEP 5: INTELLIGENT COMBINATION
    # ==========================================
    # Start with clean lines (main foundation)
    result = base_lines.copy().astype(np.float32)
    
    # Subtract shadows where needed (darker areas)
    result = result - shadows
    
    # Add texture layer (very subtle)
    result = cv2.addWeighted(result.astype(np.uint8), 0.92, texture, 0.08, 0)
    
    # Add paper grain
    result = np.clip(result.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    
    # ==========================================
    # STEP 6: FINAL ARTISTIC TOUCH
    # ==========================================
    # Gentle contrast boost
    result = cv2.convertScaleAbs(result, alpha=1.15, beta=-5)
    
    # Protect highlights (keep face bright)
    result = cv2.threshold(result, 248, 255, cv2.THRESH_TRUNC)[1]
    
    # Final normalization
    result = cv2.normalize(result, None, 5, 255, cv2.NORM_MINMAX)
    
    # Save
    cv2.imwrite(output_path, result)
    return result


if __name__ == "__main__":
    print("🎨 Creating PERFECT ARTISTIC SKETCH...")
    result = perfect_artistic_sketch("test_face.jpg", "artistic_result_PERFECT.png")
    print("✅ Saved: artistic_result_PERFECT.png")
    print("\n✨ This version has:")
    print("   • Clean, professional line art")
    print("   • Gentle volumetric shading")
    print("   • Delicate hair texture details")
    print("   • Subtle paper grain")
    print("   • Balanced brightness (not too dark!)")
