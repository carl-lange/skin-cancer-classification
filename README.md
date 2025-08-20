# Skin Cancer Classification with Deep Learning - DAT341 Assignment 5

A university coursework project implementing various CNN architectures for binary classification of dermoscopic images, comparing normalization techniques and data augmentation strategies for melanoma vs nevus detection.

This repository is an excerpt from the course project. Thus, the dataset is represented by sample images, and not all training configurations are included.
Furthermore, this repository's purpose is mainly demonstration of work, not a general, runnable machine learning system.

**Authors**: Carl Lange (carllang@chalmers.se), Pietro Rosso (pietroro@chalmers.se)  
**Course**: DAT341 - Applied Machine Learning  
**Institution**: Chalmers University of Technology

## Project Overview

Skin cancer is one of the most common types of cancer, but early detection significantly improves prognosis. This project explores different CNN architectures to automate the classification of skin lesions from dermoscopic images, potentially assisting in early screening and diagnosis.

### Key Features

- **8 Different CNN Architectures**: From baseline CNN to advanced ResNet and VGG-16 transfer learning
- **Data Augmentation Pipeline**: Horizontal flip, rotation, and random cropping to improve generalization
- **Multiple Normalization Techniques**: Batch, Group, and Layer normalization implementations
- **Transfer Learning**: Pre-trained VGG-16 with custom classification head
- **Comprehensive Evaluation**: Accuracy, F1-score, sensitivity, specificity metrics
- **Medical Context**: Optimized for high sensitivity (better to overdiagnose than miss cancer)

## 📊 Results Summary

The best performing model achieved **87.32% validation accuracy** and **86.65% test accuracy**:

| Model | Val Accuracy | Test Accuracy | Key Features |
|-------|-------------|---------------|--------------|
| **ResNet + Augmentation** | **87.32%** | **86.65%** | Skip connections + data augmentation |
| Augmented Baseline | 86.90% | Test not reported | Data augmentation only |
| VGG-16 Transfer Learning | 86.63% | Test not reported | Pre-trained ImageNet features |
| Batch Normalization CNN | 86.41% | Test not reported | Batch normalization layers |
| ResNet CNN | 85.62% | Test not reported | Residual connections |
| Group Normalization CNN | 85.29% | Test not reported | Group normalization (8 groups) |
| Baseline CNN | 84.38% | Test not reported | Simple 2-layer CNN |
| Layer Normalization CNN | 83.96% | Test not reported | Layer normalization |

### Medical Performance Metrics
- **F1 Score**: 0.8655
- **Sensitivity**: 90.92% (good at detecting melanoma)
- **Specificity**: 82.58% (some false positives)

The higher sensitivity is desirable for medical applications - it's better to have false positives that require further examination than to miss actual cancer cases.

## 🏗️ Architecture

### Models Implemented

1. **BaselineCNN**: Simple 2-layer CNN (Conv → ReLU → MaxPool × 2 + FC layers)
2. **BatchNormCNN**: Adds batch normalization for stable training
3. **GroupNormCNN**: Uses group normalization (batch-size independent)
4. **LayerNormCNN**: Applies layer normalization across spatial dimensions
5. **ResNetCNN**: Residual connections for deeper networks
6. **VGG16CNN**: Transfer learning with pre-trained VGG-16

### Data Pipeline

```
Raw Images (128×128 RGB) → Preprocessing → Augmentation → CNN → Binary Classification
```

- **Input**: 128×128 RGB dermoscopic images
- **Classes**: MEL (melanoma) = 1, NEV (nevus) = 0
- **Augmentation**: Random horizontal flip (50%), rotation (±15°), random crop
- **Normalization**: ImageNet statistics for transfer learning

## 📁 Project Structure

```
medical-image-classification/
├── src/
│   ├── models.py          # CNN model architectures
│   ├── data.py           # Data loading and preprocessing
│   └── main.py           # Training and evaluation CLI
├── data/
│   └── sample/           # Sample skin lesion images (10 images)
│       ├── mel_001.png   # Melanoma samples (5 images)
│       ├── ...
│       ├── nev_001.png   # Nevus samples (5 images)
│       └── ...
├── results/              # Training plots and evaluation results
├── models/               # Saved model checkpoints
├── pyproject.toml        # Poetry dependencies
└── README.md            # This file
```

## 🔬 Technical Details

### Model Architectures

All models follow this general structure:
- **Input**: 3×128×128 RGB images
- **Feature Extraction**: Convolutional layers with different normalization techniques
- **Classification**: Fully connected layers ending with sigmoid activation
- **Output**: Single probability score (>0.5 → MEL, ≤0.5 → NEV)

### Training Configuration

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with learning rates 0.0001-0.001
- **Batch Size**: 32-64 depending on model complexity
- **Early Stopping**: Patience of 10 epochs based on validation accuracy
- **Data Split**: 80% training, 20% validation

### Data Augmentation

The augmentation pipeline applies transformations with 50% probability:
- **Horizontal Flip**: Accounts for symmetry in skin lesions
- **Rotation**: ±15° to handle different orientations
- **Random Crop**: Improves focus on lesion features
- **Normalization**: ImageNet statistics for transfer learning

## 📈 Experimental Insights

### Key Findings

1. **Data Augmentation is Crucial**: Reduces overfitting significantly (training accuracy drops from 99%+ to ~90%, but validation accuracy improves)

2. **Normalization Techniques**: Batch normalization performs best but can overfit; group normalization is more stable

3. **Transfer Learning Works**: VGG-16 achieves competitive performance with fewer training epochs

4. **ResNet + Augmentation**: Best combination of architecture and regularization

5. **Medical Trade-offs**: Higher sensitivity prioritized over specificity for cancer detection

### Overfitting Analysis

Most models show overfitting without augmentation:
- Batch/Group/Layer norm models reach 99-100% training accuracy
- Baseline and augmented models maintain better train/val balance
- Early stopping prevents excessive overfitting

## ⚕️ Medical Context

### Clinical Relevance

This system is designed as a **screening tool** to assist dermatologists, not replace them. The high sensitivity (90.92%) means:
- Most melanomas are correctly identified
- Some benign lesions are flagged for review (false positives)
- Better safe than sorry approach for cancer detection

### Ethical Considerations

- **Bias**: Model trained primarily on lighter skin tones (dataset limitation)
- **Interpretability**: Deep learning models lack explainability for medical professionals
- **Human Oversight**: Always requires dermatologist confirmation
- **Informed Consent**: Patients should understand AI assistance is being used

### Limitations

- **Dataset Bias**: Limited to specific image conditions and skin types
- **Generalization**: Performance may vary with different camera setups or populations
- **Class Imbalance**: Real-world melanoma prevalence is much lower than 50%

## 🛠️ Development

### Adding New Models

1. Implement your model class in `src/models.py`
2. Add it to the `models_dict` in the `get_model()` function
3. Update CLI choices in `src/main.py`
4. Add model-specific configuration in `get_default_config()`

### Custom Data

Replace sample images in `data/sample/` following the naming convention:
- Melanoma: `mel_001.png`, `mel_002.png`, ...
- Nevus: `nev_001.png`, `nev_002.png`, ...

Images should be 128×128 RGB format.

## 📚 References

- **Original Paper**: Tschandl, P., et al. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images.
- **VGG-16**: Simonyan, K. & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
- **ResNet**: He, K., et al. (2016). Deep residual learning for image recognition.
- **Batch Normalization**: Ioffe, S. & Szegedy, C. (2015). Batch normalization: Accelerating deep network training.

## 📝 Citation

```bibtex
@misc{skin-cancer-classification-2024,
  title={Skin Cancer Classification with Deep Learning},
  author={Carl Lange and Pietro Rosso},
  year={2024},
  institution={Chalmers University of Technology},
  course={DAT341 Applied Machine Learning}
}
```

## 🤝 Contributing

This is an educational project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## 📄 License

Educational use only. Dataset subject to original ISIC licensing terms.

---

**⚠️ Medical Disclaimer**: This software is for educational and research purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for health-related concerns.
