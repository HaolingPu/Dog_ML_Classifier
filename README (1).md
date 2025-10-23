# Dog Breed Classification with Deep Learning

A comprehensive exploration of deep learning techniques for image classification, including CNNs, Transfer Learning, and Vision Transformers.

## 📋 Project Overview

This project builds an automated system to classify dog breeds, specifically distinguishing between Golden Retrievers and Collies. The project explores multiple deep learning approaches:

- **Convolutional Neural Networks (CNNs)** from scratch
- **Transfer Learning** using a multi-class source task
- **Vision Transformers (ViT)** architecture
- **Custom Challenge Model** design and optimization

## 🎯 Key Features

- Data preprocessing with per-channel normalization
- Custom CNN architecture implementation in PyTorch
- Transfer learning with frozen/fine-tuned layers
- Vision Transformer implementation with scaled dot-product attention
- Early stopping and model checkpointing
- Comprehensive model evaluation (Accuracy, AUROC)

## 📊 Dataset

- **Total Images**: 12,775 PNG files (3×64×64)
- **Breeds**: 10 dog breeds including Golden Retrievers, Collies, Samoyeds, Dalmatians, etc.
- **Partitions**: Training, Validation, Test, and Challenge sets
- **Target Task**: Binary classification (Collies vs. Golden Retrievers)
- **Source Task**: 8-class classification (for transfer learning)

## 🏗️ Architecture Components

### 1. Target CNN Architecture
- 3 Convolutional layers (16, 64, 8 filters)
- Max pooling layers
- Fully connected output layer
- ReLU activations
- SAME padding for spatial dimension preservation

### 2. Source CNN Architecture
- Similar to Target architecture
- Output layer adapted for 8-class classification
- Used for transfer learning pretraining

### 3. Vision Transformer (ViT)
- Patch embedding with linear projection (16 patches)
- Positional embeddings (sinusoidal)
- Learnable [CLS] token
- Multi-head self-attention (2 heads)
- 2 Transformer encoder blocks
- MLP classification head

## 🚀 Getting Started

### Prerequisites

```bash
conda create --name 445_p2 --file requirements.txt
conda activate 445_p2
```

**Required packages:**
```
imageio=2.33.1
matplotlib=3.9.2
numpy=1.26.4
pandas=2.2.2
pytorch=2.3.0
scikit-learn=1.5.1
scipy=1.13.1
torchvision=0.18.1
```

### Project Structure

```
project2/
├── data/
│   ├── dogs.csv
│   └── images/
├── model/
│   ├── target.py          # Target CNN architecture
│   ├── source.py          # Source CNN architecture
│   ├── vit.py             # Vision Transformer
│   └── challenge.py       # Challenge model
├── checkpoints/           # Model checkpoints
├── train_cnn.py          # Train target CNN
├── train_source.py       # Train source CNN
├── train_target.py       # Transfer learning training
├── train_vit.py          # Train ViT
├── train_challenge.py    # Challenge submission
├── test_cnn.py           # Evaluate CNN
├── test_vit.py           # Evaluate ViT
├── dataset.py            # Data preprocessing
├── train_common.py       # Common training functions
└── utils.py              # Helper functions
```

## 💻 Usage

### 1. Data Preprocessing

```bash
# Visualize dataset
python visualize_data.py

# Visualize label distribution
python visualize_labels.py

# Test ImageStandardizer
python test_ImageStandardizer.py
```

### 2. Train Models

```bash
# Train Target CNN (binary classification)
python train_cnn.py

# Train Source CNN (8-class classification)
python train_source.py

# Transfer Learning (freeze different layers)
python train_target.py

# Train Vision Transformer
python train_vit.py

# Train Challenge Model
python train_challenge.py
```

### 3. Evaluate Models

```bash
# Test CNN
python test_cnn.py

# Test ViT
python test_vit.py

# Generate confusion matrix
python confusion_matrix.py
```

### 4. Challenge Submission

```bash
# Generate predictions
python predict_challenge.py --uniqname=<your_uniqname>

# For GPU-trained models
python predict_challenge.py --uniqname=<your_uniqname> --cuda

# Verify output format
python test_output.py <your_uniqname>.csv
```

## 🔬 Key Implementations

### Image Standardization
Per-channel normalization is applied to all images:
- Calculate mean and standard deviation for each RGB channel from training data
- Zero-center each channel by subtracting per-channel mean
- Scale each channel by dividing by per-channel standard deviation
- Apply same statistics to validation and test sets

### Early Stopping
- Monitors validation loss during training
- Configurable patience parameter (default: 5 epochs)
- Automatically saves best model checkpoint
- Prevents overfitting and saves training time

### Transfer Learning Strategies
Explore different freezing strategies:
1. **Freeze all conv layers**: Fine-tune FC layer only
2. **Freeze first 2 conv layers**: Fine-tune last conv + FC layers
3. **Freeze first conv layer**: Fine-tune last 2 conv + FC layers
4. **Freeze no layers**: Fine-tune all layers

### Vision Transformer Components
- **Patch Embedding**: Linear projection of flattened image patches
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Scaled dot-product attention with 2 heads
- **Transformer Encoder**: 2 blocks with LayerNorm and MLP
- **Classification Head**: Extract [CLS] token output and apply MLP

## 📈 Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **AUROC**: Area Under Receiver Operating Characteristic curve
- **Confusion Matrix**: Per-class performance visualization

### Training Hyperparameters

**Target CNN:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Patience: 5

**Source CNN:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3, weight_decay=0.01)
- Batch size: 64
- Patience: 10

**Vision Transformer:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Embedding dim: 16
- Attention heads: 2
- Transformer blocks: 2

## 🎓 Learning Outcomes

This project demonstrates:
1. Implementing CNNs from scratch in PyTorch
2. Understanding transfer learning principles and applications
3. Working with Vision Transformers and attention mechanisms
4. Applying regularization techniques (weight decay, dropout)
5. Performing hyperparameter tuning and model selection
6. Comprehensive model evaluation and comparison

## 📝 Implementation Highlights

### Custom CNN Architecture
```python
# Example architecture flow
Input (3×64×64)
→ Conv1 (16 filters, 5×5) + ReLU + MaxPool
→ Conv2 (64 filters, 5×5) + ReLU + MaxPool
→ Conv3 (8 filters, 5×5) + ReLU
→ Flatten
→ FC (2 outputs)
```

### Scaled Dot-Product Attention
```python
Attention(Q, K, V) = softmax(QK^T / √d) V
```

### Transfer Learning Workflow
```
Source Task (8 classes) → Pretrain CNN
                       ↓
              Transfer weights
                       ↓
Target Task (2 classes) → Fine-tune with frozen layers
```

## ⚠️ Important Notes

- **No pre-trained models**: All models must be trained from scratch
- **No external data**: Use only the provided dataset
- **Data augmentation**: Allowed for challenge submission
- **GPU usage**: Optional, but specify in submission
- **Checkpointing**: Models automatically save best epoch

## 🔗 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2021
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Conv Arithmetic Guide](https://github.com/vdumoulin/conv_arithmetic) - Vincent Dumoulin

## 📄 License

This project is part of an academic course assignment. Please follow your institution's academic integrity policies when using this code.

## 🤝 Contributing

This is an academic project. If you're working on a similar assignment:
- Use this as a learning reference only
- Implement your own solutions
- Follow academic integrity guidelines
- Cite appropriately if inspired by this work

---

**Tech Stack**: PyTorch • NumPy • Pandas • Matplotlib • scikit-learn

**Topics**: Deep Learning • Computer Vision • CNNs • Transfer Learning • Vision Transformers • Image Classification
