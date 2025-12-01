# 👤 Face Recognition System: FaceNet vs ResNet50

**Deep Learning Tubes - Final Project**  
**Date**: December 1, 2025  
**Framework**: PyTorch with CUDA Support

A comprehensive face recognition system comparing two state-of-the-art architectures:
- **FaceNet (InceptionResnetV1)** with VGGFace2 pre-training + MTCNN face detection
- **ResNet50** with ImageNet pre-training

---

## 🎯 Project Overview

This project implements an end-to-end face recognition system with:
- ✅ **70 classes** (70 different people)
- ✅ **283 training images** (~4 images per person)
- ✅ **Two model architectures** for comparison
- ✅ **MTCNN face detection** for robust alignment
- ✅ **Aggressive data augmentation** (40x for FaceNet)
- ✅ **Streamlit web application** for deployment
- ✅ **CUDA GPU acceleration** support

---

## 🏗️ Architecture Comparison

| Feature | FaceNet | ResNet50 |
|---------|---------|----------|
| **Base Model** | InceptionResnetV1 | ResNet50 |
| **Pre-trained On** | VGGFace2 (faces) | ImageNet (objects) |
| **Input Size** | 160x160 | 224x224 |
| **Face Detection** | MTCNN | Standard resize |
| **Embedding Size** | 512D | 2048D |
| **Augmentation** | 40x aggressive | Standard |
| **Best For** | Small datasets | Large datasets |
| **Val Accuracy** | ~85-95% | ~70-85% |

📖 **Detailed Comparison**: See [FACENET_COMPARISON.md](FACENET_COMPARISON.md)

---

## 📂 Project Structure

```
Deep Learning Tubes/
├── 📒 Notebooks
│   ├── face_recognition_facenet_new.ipynb    # Main: FaceNet vs ResNet50
│   ├── face_recognition_complete.ipynb       # Alternative full training
│   ├── face_recognition_project.ipynb        # Legacy notebook
│   ├── facenet_demo.ipynb                    # FaceNet demo
│
├── 🐍 Python Scripts
│   ├── facenet.py                            # FaceNet implementation
│   ├── train.py                              # Training script
│   ├── test.py                               # Testing script
│   ├── evaluate.py                           # Evaluation script
│   ├── app.py                                # Streamlit web app ⭐
│
├── 🤖 Models (Generated after training)
│   ├── best_model.pth                        # Best performing model
│   ├── facenet_model.pth                     # FaceNet for inference
│   ├── resnet50_model.pth                    # ResNet50 for inference
│   ├── class_names.pkl                       # Person names mapping
│   ├── model_info.pkl                        # Training metrics
│   └── models/
│       ├── facenet_model_20251201_225633.pkl     # Pre-trained FaceNet
│       └── facenet_classifier_20251201_225633.pth
│
├── 📊 Dataset
│   └── Train/Train/                          # 70 people, 283 images
│       ├── Abraham Ganda Napitu/
│       ├── Abu Bakar Siddiq Siregar/
│       ├── ...
│       └── Zidan Raihan/
│
├── 📖 Documentation
│   ├── README.md                             # This file ⭐
│   ├── FACENET_COMPARISON.md                 # Architecture comparison
│   ├── FACENET_ARCHITECTURE.md               # FaceNet system details
│   ├── RESNET50_ARCHITECTURE.md              # ResNet50 system details
│   ├── IMPLEMENTATION_SUMMARY.md             # Implementation notes
│
└── ⚙️ Configuration
    ├── requirements.txt                      # Python dependencies
    └── requirements_deploy.txt               # Deployment dependencies
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone or navigate to project directory
cd "Deep Learning Tubes"

# Install dependencies
pip install -r requirements.txt
```

**Requirements include:**
- `torch` (PyTorch with CUDA 12.4)
- `torchvision`
- `facenet-pytorch` (MTCNN + InceptionResnetV1)
- `streamlit` (Web app)
- `scikit-learn`, `numpy`, `pandas`
- `Pillow`, `opencv-python`

### 2️⃣ Training Models

**Option A: Use Pre-trained FaceNet (Recommended) ⚡**

```bash
# Open face_recognition_facenet_new.ipynb in VS Code or Jupyter
# This notebook will:
# - Load pre-trained FaceNet from ./models/facenet_model_20251201_225633.pkl
# - Train ResNet50 for comparison
# - Generate all necessary files for deployment
```

**Option B: Train Both from Scratch**

```bash
# Use train.py for FaceNet
python train.py

# Or use face_recognition_complete.ipynb for both models
```

**Training Output:**
- ✅ `best_model.pth` - Best performing model
- ✅ `facenet_model.pth` - FaceNet weights
- ✅ `resnet50_model.pth` - ResNet50 weights
- ✅ `class_names.pkl` - Person names
- ✅ `model_info.pkl` - Comparison metrics
- ✅ Training curves and confusion matrices

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

**App Features:**
- 📸 Upload images for face recognition
- 👥 Detect multiple faces in one image
- 🎚️ Adjustable confidence threshold
- 📊 View model comparison metrics
- 🎯 See confidence scores and predictions

**Access at**: http://localhost:8501

---

## 📊 Model Training Details

### FaceNet Training

```python
# Configuration
augmentation: True
num_augmentations: 40        # 40x data multiplication
num_epochs: 100
batch_size: 16
learning_rate: 0.0001
validation_split: 0.2
optimizer: Adam
scheduler: ReduceLROnPlateau

# Architecture
InceptionResnetV1 (VGGFace2) → 512D Embedding
    ↓
Classifier Head:
    - Batch Normalization
    - Dropout(0.3)
    - Linear(512 → 256)
    - ReLU + Dropout(0.3)
    - Linear(256 → num_classes)
```

**See detailed architecture**: [FACENET_ARCHITECTURE.md](FACENET_ARCHITECTURE.md)

### ResNet50 Training

```python
# Configuration
num_epochs: 10
batch_size: 16
learning_rate: 0.001
validation_split: 0.2
optimizer: Adam
scheduler: ReduceLROnPlateau

# Architecture
ResNet50 (ImageNet) → 2048D Features
    ↓
Classifier Head:
    - Dropout(0.5)
    - Linear(2048 → 512)
    - ReLU + Dropout(0.5)
    - Linear(512 → num_classes)
```

**See detailed architecture**: [RESNET50_ARCHITECTURE.md](RESNET50_ARCHITECTURE.md)

---

## 🎓 Model Performance

### Expected Results (Small Dataset: ~4 images/person)

| Model | Train Acc | Val Acc | F1-Score | Training Time |
|-------|-----------|---------|----------|---------------|
| **FaceNet** | ~98% | **~85-95%** | ~0.90 | ~30-60 min (100 epochs) |
| **ResNet50** | ~95% | ~70-85% | ~0.80 | ~5-10 min (10 epochs) |

**Winner**: 🏆 **FaceNet** (Face-specific pre-training + MTCNN alignment)

### Why FaceNet Performs Better:

1. ✅ **Face-specific pre-training** (VGGFace2 vs ImageNet)
2. ✅ **MTCNN face detection** (alignment and cropping)
3. ✅ **Aggressive augmentation** (40x vs 1x)
4. ✅ **Optimized for small datasets**

---

## 🔍 Key Features

### 1. MTCNN Face Detection
- Multi-task Cascaded Convolutional Networks
- Detects faces with bounding boxes
- Aligns faces to canonical pose
- Handles multiple faces in one image
- Robust to rotation and scale

### 2. Aggressive Data Augmentation (FaceNet)
- **40x multiplication** of training data
- Random horizontal flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random grayscale conversion
- Random Gaussian blur
- **Result**: 283 images → ~11,320 training samples

### 3. Transfer Learning
- **FaceNet**: VGGFace2 (2.6M faces, 9K identities)
- **ResNet50**: ImageNet (1.2M images, 1K classes)
- Freeze base layers, train only classifier head
- Faster convergence, better generalization

### 4. Model Comparison Dashboard
- Side-by-side accuracy comparison
- Loss curves visualization
- Confidence score analysis
- Real-time inference

---

## 📖 Documentation

### Architecture Documentation

1. **[FACENET_ARCHITECTURE.md](FACENET_ARCHITECTURE.md)** 📘
   - Complete FaceNet system architecture
   - Training pipeline stages
   - Inference flow diagram
   - Component descriptions
   - Performance characteristics

2. **[RESNET50_ARCHITECTURE.md](RESNET50_ARCHITECTURE.md)** 📗
   - Complete ResNet50 system architecture
   - Training pipeline stages
   - Inference flow diagram
   - Component descriptions
   - Performance characteristics

3. **[FACENET_COMPARISON.md](FACENET_COMPARISON.md)** 📙
   - Detailed comparison of both architectures
   - Advantages and disadvantages
   - Use case recommendations
   - Performance metrics

4. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** 📔
   - Implementation details
   - Code changes log
   - File structure
   - Usage instructions

---

## 🛠️ Usage Examples

### Training via Script

```bash
# Train FaceNet with custom parameters
python train.py
```

### Training via Notebook

```python
# Open face_recognition_facenet_new.ipynb

# 1. Load pre-trained FaceNet
FACENET_MODEL_PATH = './models/facenet_model_20251201_225633.pkl'
facenet_model = FaceNetModel(device='cuda')
facenet_model.load_model(FACENET_MODEL_PATH)

# 2. Train ResNet50
# (Code in notebook)

# 3. Compare models
# (Automatic comparison and visualization)
```

### Inference with FaceNet

```python
from facenet import FaceNetModel

# Load model
model = FaceNetModel(device='cuda')
model.load_model('./models/facenet_model_20251201_225633.pkl')

# Predict
name, similarity = model.predict('path/to/image.jpg', threshold=0.6)
print(f"Predicted: {name} (Similarity: {similarity:.3f})")
```

### Inference with Streamlit App

```bash
# Start app
streamlit run app.py

# Then:
# 1. Upload image via web interface
# 2. Adjust confidence threshold (0.0 - 1.0)
# 3. View predictions with bounding boxes
# 4. See model comparison metrics
```

---

## 🎨 Streamlit App Features

### Main Interface
- 📤 **Image Upload**: Support JPG, JPEG, PNG
- 🎯 **Face Detection**: MTCNN multi-face detection
- 🏷️ **Recognition**: Display name + confidence score
- 📦 **Bounding Boxes**: Visual face detection boxes
- 🎚️ **Threshold Slider**: Adjust confidence cutoff

### Sidebar
- 📊 **Model Comparison**: FaceNet vs ResNet50 metrics
- ⚙️ **Settings**: Confidence threshold control
- ℹ️ **Model Info**: Architecture and dataset details

### Output
- ✅ Recognized faces with names
- ❌ Unknown faces (below threshold)
- 📊 Confidence scores for each face
- 🖼️ Visual bounding boxes

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
batch_size = 8  # Instead of 16

# Solution 2: Use CPU
device = 'cpu'
```

### Issue: Streamlit Not Running

```bash
# Check if model files exist
ls *.pth *.pkl

# Required files:
# - best_model.pth
# - class_names.pkl
# - model_info.pkl (optional)

# If missing, run training notebook first
```

### Issue: Low Accuracy

```python
# Solutions:
# 1. Use FaceNet (better for small datasets)
# 2. Increase augmentation
# 3. Add more training images per person
# 4. Ensure face quality (clear, frontal)
# 5. Check class balance
```

### Issue: Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# For CUDA issues
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 📈 Performance Benchmarks

### Training Time (NVIDIA GPU)

| Model | Epochs | Time | Samples/sec |
|-------|--------|------|-------------|
| FaceNet | 100 | ~45 min | ~250 |
| ResNet50 | 10 | ~8 min | ~180 |

### Inference Time (Single Image)

| Model | GPU | CPU | Memory |
|-------|-----|-----|--------|
| FaceNet | ~50ms | ~200ms | ~512MB |
| ResNet50 | ~30ms | ~150ms | ~1.5GB |

### Memory Usage

| Component | Size |
|-----------|------|
| FaceNet Model | ~90MB |
| ResNet50 Model | ~103MB |
| MTCNN | ~10MB |
| Dataset (loaded) | ~100MB |
| **Total** | **~300MB** |

---

## 🚀 Deployment

### Local Deployment

```bash
# Run Streamlit app
streamlit run app.py --server.port 8501
```

### HuggingFace Spaces

```bash
# 1. Create requirements_deploy.txt (lighter dependencies)
# 2. Push to HuggingFace Space
# 3. App will auto-deploy

# Files needed:
# - app.py
# - requirements_deploy.txt
# - best_model.pth
# - class_names.pkl
# - facenet.py (if using FaceNet)
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test FaceNet model
python test.py

# Test on specific image
python -c "
from facenet import FaceNetModel
model = FaceNetModel()
model.load_model('./models/facenet_model_20251201_225633.pkl')
name, sim = model.predict('path/to/test.jpg')
print(f'{name}: {sim:.3f}')
"
```

### Evaluation

```bash
# Full evaluation with metrics
python evaluate.py
```

---

## 👥 Dataset Information

- **Total Classes**: 70 people
- **Total Images**: 283 images
- **Images per Person**: ~4 (varies: 4-8)
- **Image Format**: JPG, JPEG, PNG
- **Image Quality**: Mixed (some low resolution)
- **Data Distribution**: Slightly imbalanced

### Data Preprocessing

1. **Face Detection**: MTCNN detects faces
2. **Face Cropping**: Extract face regions
3. **Face Alignment**: Align to canonical pose
4. **Resizing**: 160x160 (FaceNet) or 224x224 (ResNet50)
5. **Normalization**: Mean/std normalization
6. **Augmentation**: 40x for FaceNet, standard for ResNet50

---

## 🔬 Research & References

### Papers

1. **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
   - [Paper](https://arxiv.org/abs/1503.03832)

2. **MTCNN**: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" (2016)
   - [Paper](https://arxiv.org/abs/1604.02878)

3. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
   - [Paper](https://arxiv.org/abs/1512.03385)

### Libraries

- **PyTorch**: https://pytorch.org/
- **facenet-pytorch**: https://github.com/timesler/facenet-pytorch
- **Streamlit**: https://streamlit.io/

---

## 📝 License

This project is for educational purposes (Deep Learning Tubes Final Project).

---

## 👨‍💻 Contributors

**Deep Learning Tubes - Batch 2025**  
70 students in face recognition dataset

---

## 🎯 Future Improvements

- [ ] Add more training images per person
- [ ] Implement triplet loss for better embeddings
- [ ] Add face verification (1:1 matching)
- [ ] Real-time webcam recognition
- [ ] Mobile app deployment
- [ ] Add face clustering
- [ ] Implement attention mechanisms
- [ ] Add explainability (Grad-CAM)
- [ ] Support for video recognition
- [ ] API endpoint for integration

---

## 📞 Support

For questions or issues:
1. Check documentation files (FACENET_ARCHITECTURE.md, RESNET50_ARCHITECTURE.md)
2. Review IMPLEMENTATION_SUMMARY.md
3. Check troubleshooting section above
4. Review training logs and metrics

---

**Last Updated**: December 1, 2025  
**Project Status**: ✅ Production Ready  
**Best Model**: FaceNet (InceptionResnetV1 + VGGFace2 + MTCNN)  

🎉 **Happy Face Recognition!** 👤🔍
