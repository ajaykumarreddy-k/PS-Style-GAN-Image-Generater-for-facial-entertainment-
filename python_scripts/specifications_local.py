import torch
print(f"🔥 PyTorch Version: {torch.__version__}")
print(f"💻 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ WARNING: Using CPU. Training will be too slow.")