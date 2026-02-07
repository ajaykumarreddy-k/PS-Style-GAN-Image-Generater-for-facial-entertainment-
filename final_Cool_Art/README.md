# 🎨 final_Cool_Art - Production Deployment

> **Note:** For complete project documentation, see the [main README.md](../README.md) in the parent directory.

This folder contains the **self-contained production system** for the hybrid sketch generator.

## 🚀 Quick Start

### Web Interface (Recommended)

```bash
cd final_Cool_Art
source ../.venv/bin/activate
python gradio_app.py
```

Open browser: **http://localhost:7860**

### Batch Processing

```bash
python enhanced_hybrid.py
```

Generates 3 variations at 30%, 35%, 40% AI strength.

## 📦 Contents

| File | Description |
|------|-------------|
| `gradio_app.py` | Production web interface |
| `enhanced_hybrid.py` | Batch processing script |
| `final_model_SHADING.pth` | Trained AI model (46 MB) |
| `test_face.jpg` | Sample input |
| `hybrid_result_ENHANCED_35.png` | Example output |
| `requirements.txt` | Dependencies |

## ⚙️ Configuration (ENHANCED_35)

- **Blur Kernel**: 18 (sharp lines)
- **AI Strength**: 35% (optimal balance)
- **Preprocessing**: Histogram eq. + sharpening
- **Contrast**: AI 1.15x, Final 1.05x

## 🔧 Extending

This folder is designed for easy extension:

1. **Model Swapping**: Replace `.pth` file
2. **Pipeline Tuning**: Edit `gradio_app.py` processing function
3. **UI Customization**: Modify Gradio layout/CSS
4. **New Presets**: Add preset buttons
5. **API Integration**: Wrap in Flask/FastAPI

## 📚 Full Documentation

See [../README.md](../README.md) for:
- Complete architecture
- Methodology & training
- Performance metrics
- Comparative analysis
- Extension guides

---

**Ready to create amazing sketches!** 🎨
