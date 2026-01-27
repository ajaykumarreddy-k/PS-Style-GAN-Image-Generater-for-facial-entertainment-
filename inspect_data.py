import os
import pandas as pd
import glob
import zipfile

# CONFIG
ZIP_NAME = "SKSF-A.zip"
EXTRACT_DIR = "./dataset_extracted"

def inspect():
    print("🔍 DIAGNOSTIC MODE: checking paths...")
    
    # 1. Check if Extracted
    if not os.path.exists(EXTRACT_DIR):
        print(f"❌ Extraction folder '{EXTRACT_DIR}' not found. Did the previous script run?")
        if os.path.exists(ZIP_NAME):
            print(f"📦 Found zip '{ZIP_NAME}'. Extracting now...")
            with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
        else:
            print("❌ Zip file also missing! Please put SKSF-A.zip here.")
            return

    # 2. Find CSV
    csv_files = glob.glob(f"{EXTRACT_DIR}/**/*.csv", recursive=True)
    if not csv_files:
        print("❌ No CSV file found in the extracted folder.")
        return
    
    target_csv = csv_files[0]
    print(f"📄 Found CSV: {target_csv}")
    
    # 3. Find Photo Folder
    # Look for any folder named 'photos' or 'images'
    photo_folders = glob.glob(f"{EXTRACT_DIR}/**/photos", recursive=True)
    if not photo_folders:
        print("❌ No 'photos' folder found!")
        print("   Current folders exist:")
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for d in dirs:
                print(f"    - {os.path.join(root, d)}")
        return

    photo_dir = photo_folders[0]
    print(f"📂 Found Photo Dir: {photo_dir}")
    
    # 4. List First 3 Actual Files
    actual_files = os.listdir(photo_dir)[:3]
    print(f"   👉 Actual files on disk: {actual_files}")

    # 5. Read CSV and Compare
    try:
        df = pd.read_csv(target_csv)
        print(f"   👉 CSV content (first row): {df.iloc[0].values.tolist()}")
        
        # TEST MATCH
        csv_val = str(df.iloc[0, 0]) # Assuming col 0 is filename
        test_path = os.path.join(photo_dir, csv_val)
        
        print("\n--- MATCH TEST ---")
        print(f"1. CSV says file is: '{csv_val}'")
        print(f"2. Script looks at:  '{test_path}'")
        
        if os.path.exists(test_path):
            print("✅ EXISTS! (Direct match)")
        elif os.path.exists(test_path + ".jpg"):
             print(f"✅ EXISTS! (Found with .jpg extension: {test_path}.jpg)")
        elif os.path.exists(test_path + ".png"):
             print(f"✅ EXISTS! (Found with .png extension: {test_path}.png)")
        else:
            print("❌ FILE NOT FOUND. This is why the training crashes.")
            print("   Possible reasons: Padding (6 vs 006), Extension missing, or wrong column.")

    except Exception as e:
        print(f"❌ Could not read CSV: {e}")

if __name__ == "__main__":
    inspect()