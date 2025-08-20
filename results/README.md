# Results

This directory contains training results and evaluation plots for the skin cancer classification models.

## Expected Files

When you run training experiments, the following files will be generated:

- `training_plots_{model_name}.png` - Training and validation accuracy/loss curves
- `confusion_matrix_{model_name}.png` - Confusion matrices for test set evaluation  
- `model_comparison.png` - Comparison of different model performances

## Key Results from Original Assignment

According to the lab report, the following results were achieved:

### Model Performance (Validation Accuracy)

| Model | Validation Accuracy | Training Accuracy |
|-------|-------------------|------------------|
| Baseline | 84.38% | 88.88% |
| Batch Norm | 86.41% | 100.00% |
| Group Norm | 85.29% | 99.65% |
| Layer Norm | 83.96% | 99.47% |
| Augmented | 86.90% | 88.63% |
| ResNet | 85.62% | 99.00% |
| VGG-16 | 86.63% | 99.71% |
| **ResNet + Augmented** | **87.32%** | **90.36%** |

### Best Model: ResNet + Augmented

- **Test Accuracy**: 86.65%
- **F1 Score**: 0.8655
- **Sensitivity**: 0.9092 (good at detecting melanoma)
- **Specificity**: 0.8258

### Key Findings

1. **Data augmentation helps prevent overfitting** - Models with augmentation show smaller gaps between training and validation accuracy
2. **Batch normalization performs well** but can lead to overfitting
3. **ResNet with residual connections** combined with augmentation achieves the best performance
4. **VGG-16 transfer learning** shows competitive performance
5. **Higher sensitivity than specificity** is desirable for medical diagnosis (better to have false positives than false negatives)
