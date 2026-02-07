import cv2
import numpy as np

def refined_artistic_sketch(image_path, output_path):
    """
    REFINED ARTISTIC SKETCH
    Now with clean jaw, crisp hair edges, and clear background
    """
    
    # Load and prepare
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ==========================================
    # STEP 1: SMART BACKGROUND SEPARATION
    # ==========================================
    # Detect edges to separate subject from background
    edges = cv2.Canny(gray, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Create subject mask using GrabCut-like approach
    # Simple thresholding + morphology for mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Blur mask edges for smooth transition
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # ==========================================
    # STEP 2: ENHANCED BASE LINE ART
    # ==========================================
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    inverted = 255 - enhanced
    
    # Multi-scale color dodge with edge awareness
    blur1 = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch_soft = cv2.divide(enhanced, 255 - blur1, scale=256)
    
    blur2 = cv2.GaussianBlur(inverted, (9, 9), 0)
    sketch_detail = cv2.divide(enhanced, 255 - blur2, scale=256)
    
    # Blend with more detail preservation
    base_lines = cv2.addWeighted(sketch_soft, 0.6, sketch_detail, 0.4, 0)
    
    # ==========================================
    # STEP 3: CLEAN BACKGROUND
    # ==========================================
    # Make background pure white (no sketch lines)
    background_cleaned = np.ones_like(base_lines) * 255
    
    # Blend foreground and background using mask
    mask_3d = mask_blur[:, :, np.newaxis] / 255.0
    result = (base_lines * mask_3d + background_cleaned * (1 - mask_3d)).astype(np.uint8)
    
    # ==========================================
    # STEP 4: REFINED FACIAL CONTOURS (JAW FIX)
    # ==========================================
    # Bilateral filter for smooth jaw/face lines
    smooth = cv2.bilateralFilter(gray, 9, 150, 150)
    
    # Gentle morphological gradient for contours
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    contours = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, kernel_small)
    contours = cv2.GaussianBlur(contours, (15, 15), 0)
    
    # Normalize and invert
    contours = cv2.normalize(contours, None, 0, 40, cv2.NORM_MINMAX)
    contours_inv = 255 - contours
    
    # Apply only to face area (not background)
    contours_masked = (contours_inv * mask_3d[:, :, 0] + 255 * (1 - mask_3d[:, :, 0])).astype(np.uint8)
    
    # Blend with main result
    result = cv2.addWeighted(result, 0.85, contours_masked, 0.15, 0)
    
    # ==========================================
    # STEP 5: HAIR DETAIL ENHANCEMENT
    # ==========================================
    # Use Sobel for directional hair strands
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient = np.uint8(gradient / gradient.max() * 255)
    
    # Create hair texture (subtle)
    hair_texture = 255 - gradient
    hair_texture = cv2.GaussianBlur(hair_texture, (3, 3), 0)
    
    # Apply only where there's already detail (hair regions)
    detail_mask = cv2.threshold(base_lines, 240, 255, cv2.THRESH_BINARY_INV)[1]
    hair_masked = cv2.bitwise_and(hair_texture, hair_texture, mask=detail_mask)
    
    # Blend hair detail
    result = cv2.addWeighted(result, 0.94, hair_masked, 0.06, 0)
    
    # ==========================================
    # STEP 6: SUBTLE PAPER GRAIN (FOREGROUND ONLY)
    # ==========================================
    h, w = gray.shape
    grain = np.random.normal(0, 2, (h, w)).astype(np.int16)
    
    # Apply grain only to foreground
    grain_masked = grain * mask_3d[:, :, 0]
    result = np.clip(result.astype(np.int16) + grain_masked, 0, 255).astype(np.uint8)
    
    # ==========================================
    # STEP 7: FINAL POLISH
    # ==========================================
    # Gentle contrast
    result = cv2.convertScaleAbs(result, alpha=1.12, beta=-3)
    
    # Protect whites
    result = cv2.threshold(result, 250, 255, cv2.THRESH_TRUNC)[1]
    
    # Final normalization
    result = cv2.normalize(result, None, 10, 255, cv2.NORM_MINMAX)
    
    # Save
    cv2.imwrite(output_path, result)
    return result


if __name__ == "__main__":
    print("🎨 Creating REFINED artistic sketch...")
    print("   Fixing: Jaw lines, hair edges, background cleanup")
    
    result = refined_artistic_sketch("test_face.jpg", "artistic_result_REFINED.png")
    
    print("✅ Saved: artistic_result_REFINED.png")
    print("\n✨ Improvements:")
    print("   • Clean white background (no stray lines)")
    print("   • Smooth jaw and facial contours")
    print("   • Crisp hair edges with detail")
    print("   • Better foreground/background separation")
