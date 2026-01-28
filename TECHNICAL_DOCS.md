This is the specific, high-level technical documentation for your project report. It focuses on the **architecture, challenges, and final deliverables**. You can use this content for your project report, presentation slides, or a `TECHNICAL_DOCS.md` file.

---

# 📘 Project Vibe: Technical Documentation

## 1. Final Deliverables (The "Golden" Files)

These are the two critical components that power the final application.

* **🏆 The Champion Model File:** `final_model_SHADING.pth`
* **Role:** The "Brain."
* **Description:** This is the PyTorch Generator model weights saved after the final training run.
* **Why it was chosen:** Unlike previous iterations, this model was trained with **Gray Mass Loss** and **Cutout Augmentation**, allowing it to understand density and shading rather than just binary outlines.


* **🚀 The Production Inference Script:** `run_inference.py`
* **Role:** The "Engine."
* **Description:** This is the Python script that loads the model and processes the image.
* **Key Function:** It contains the **Layered Synthesis Logic**—it mathematically separates the facial structure from the textural details and applies masked noise to simulate graphite physics.



---

## 2. File Manifest (What Everything Does)

### **Core System**

* **`run_inference.py`**: The main executable. It takes a raw input image (`test_face.jpg`), passes it through the champion model, applies post-processing, and saves the final result (`result_FINAL_POLISH.png`).
* **`app.py`**: The User Interface. A Streamlit/Gradio-based web application that allows users to interact with the model via a browser.

### **Training & Development**

* **`stylesketch_implementation.py`**: The *Baseline* training script. This was the first attempt using a standard GAN, which produced stiff, binary results.
* **`stylesketch_shading.py`**: The *Advanced* training script. This version introduced **Pixel-Wise Weighting** and **Cutout Augmentation** to solve the ghosting and stencil artifacts.
* **`specifications_local.py`**: The configuration file hosting hyperparameters (Learning Rate, Batch Size, Epochs).
* **`inspect_data.py`**: A diagnostic utility created to scan the dataset for corrupted images and verify label integrity before training.

### **Artifacts**

* **`dataset_extracted/`**: The directory containing the raw training pairs (Photo  Sketch).
* **`output_shading/`**: The directory where training logs and intermediate model checkpoints were stored.

---

## 3. Technical Challenges & Resolutions (The Engineering Journey)

The project faced four distinct technical hurdles. Here is the chronological resolution path:

### **Phase 1: Dataset Integrity Failure**

* **🔴 The Difficulty:** The initial dataset contained corrupted image headers and mismatched pairs. The training loop crashed immediately upon data loading.
* **🟢 The Resolution:** Developed a custom diagnostic tool (`inspect_data.py`) to perform a "health check" on every file. We filtered out unreadable bytes and enforced strict pair matching, creating a clean, error-free dataset.

### **Phase 2: The "Stencil" Artifact**

* **🔴 The Difficulty:** Early models produced sketches that looked like "black stamps" or binary stencils. They lacked gray tones, pressure variance, and subtle shading.
* **🟢 The Resolution:** Implemented **Cutout Augmentation**. By randomly masking parts of the input during training, we forced the model to "hallucinate" structure and learn global coherence. We also adopted **Perceptual Loss (VGG-19)** to prioritize artistic style over pixel-perfect accuracy.

### **Phase 3: The "Wobbly Line" & Aliasing**

* **🔴 The Difficulty:** When upscaling the GAN output to HD, structural lines (like the jawline) became jagged ("staircase effect") or wavy. Standard sharpening filters introduced dirty artifacts.
* **🟢 The Resolution:** Engineered a **Layered Synthesis Pipeline**.
* **Layer A (Structure):** Used high-threshold binarization to isolate high-confidence features (Jaw, Eyes).
* **Layer B (Detail):** Used low-threshold gamma correction to capture low-confidence features (Hair, Shading).
* **Merge:** Combined them using a mathematical `minimum` blend to keep strong lines solid and weak lines soft.



### **Phase 4: The "Digital" Artificiality**

* **🔴 The Difficulty:** The result was mathematically perfect but "soulless." It looked like a vector illustration or coloring book page, lacking the physical grain of a real pencil sketch.
* **🟢 The Resolution:** Implemented **Masked Stochastic Noise**.
* Instead of adding global noise (which makes the image look like a bad scan), we calculated a mask of the pencil strokes.
* We injected noise **only** inside those strokes, simulating the physical interaction of graphite texture on paper while keeping the background pure white.



---

## 4. Final System Architecture

**Input (RGB Photo)  Generator (`.pth`)  [Layer Separation]  [Noise Injection]  Final Output**

1. **Prediction:** The GAN predicts the raw sketch structure.
2. **Decomposition:** The system splits the image into **Geometric Bones** vs. **Textural Details**.
3. **Physics Simulation:** Granular noise is injected into dark regions to mimic graphite density.
4. **Composition:** Layers are merged to produce a professional, high-definition pencil sketch.