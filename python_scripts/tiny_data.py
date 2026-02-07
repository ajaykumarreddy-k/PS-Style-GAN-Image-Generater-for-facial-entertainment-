import os
import zipfile
import random

# CONFIGURATION
ROOT_DIR = "dataset_extracted"
OUTPUT_ZIP = "tiny_dataset.zip"
NUM_PAIRS = 50

def find_image_folders(root_path):
    """Recursively finds folders that contain images."""
    image_folders = []
    for root, dirs, files in os.walk(root_path):
        # Count images in this specific folder
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) > 0:
            image_folders.append((root, len(images)))
    
    # Sort by count (largest folders first)
    image_folders.sort(key=lambda x: x[1], reverse=True)
    return image_folders

def create_smart_zip():
    print(f"🔍 Scanning '{ROOT_DIR}' for image folders...")
    
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Error: Folder '{ROOT_DIR}' does not exist!")
        return

    folders = find_image_folders(ROOT_DIR)
    
    if len(folders) < 2:
        print("❌ Error: Could not find two distinct folders with images (Need Photos & Sketches).")
        print("Found only:", folders)
        return

    # Assume the two largest folders are the Photo/Sketch pair
    folder_a = folders[0][0] # Largest folder
    folder_b = folders[1][0] # Second largest
    
    print(f"✅ Found potential data folders:")
    print(f"   Folder A (Photos?):  {folder_a} ({folders[0][1]} images)")
    print(f"   Folder B (Sketches?): {folder_b} ({folders[1][1]} images)")
    
    # Get list of files in A
    files_a = [f for f in os.listdir(folder_a) if f.lower().endswith(('.jpg', '.png'))]
    
    # Find matches in B (files with same name)
    pairs = []
    for f in files_a:
        if os.path.exists(os.path.join(folder_b, f)):
            pairs.append(f)
            
    print(f"🔗 Found {len(pairs)} matching pairs between them.")
    
    if len(pairs) == 0:
        print("⚠️ No matching filenames found! Trying to pair by index just to get it working...")
        # Fallback: Just zip random files if names don't match
        files_b = [f for f in os.listdir(folder_b) if f.lower().endswith(('.jpg', '.png'))]
        pairs = list(zip(files_a, files_b))

    # Select random 50
    selected_pairs = random.sample(pairs, min(NUM_PAIRS, len(pairs)))
    
    print(f"📦 Zipping {len(selected_pairs)} pairs into {OUTPUT_ZIP}...")

    with zipfile.ZipFile(OUTPUT_ZIP, 'w') as zipf:
        for p in selected_pairs:
            # Handle if p is a tuple (mismatched names) or string (matched names)
            if isinstance(p, tuple):
                name_a, name_b = p
                arc_name = name_a # Use name form A for zip
            else:
                name_a = name_b = arc_name = p
            
            # Write Photo
            zipf.write(os.path.join(folder_a, name_a), arcname=f"train_A/{arc_name}")
            # Write Sketch
            zipf.write(os.path.join(folder_b, name_b), arcname=f"train_B/{arc_name}")

    print(f"🏆 SUCCESS! Created '{OUTPUT_ZIP}'. Upload this to Kaggle.")

if __name__ == "__main__":
    create_smart_zip()