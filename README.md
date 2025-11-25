# Iris Flower Classification

A complete Machine Learning project for classifying iris flowers using scikit-learn. This project implements multiple classification algorithms, performs data exploration, and provides detailed model evaluation metrics.

## Project Overview

This project demonstrates a full machine learning workflow including:
- Data loading and exploration
- Data visualization and analysis
- Feature scaling and normalization
- Model training with multiple algorithms
- Model evaluation and comparison
- Performance visualization

## Dataset

The **Iris Dataset** contains 150 samples with 4 features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Target Classes:** 3 iris species (Setosa, Versicolor, Virginica)

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## Usage

Run the classification script:
```bash
python iris_classification.py
```

## Project Structure

```
iris-flower-classification/
├── iris_classification.py      # Main Python script
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

## Implementation Details

### Models Used

1. **Logistic Regression**
   - Algorithm: Linear classifier
   - Activation: Softmax (for multiclass)
   - Training: Scaled features
   - Best for: Quick, interpretable results

2. **Decision Tree Classifier**
   - Max Depth: 5 (to prevent overfitting)
   - Training: Original features (no scaling needed)
   - Best for: Non-linear relationships

### Data Preparation

- **Train/Test Split:** 70/30
- **Feature Scaling:** StandardScaler (for Logistic Regression)
- **Class Distribution:** Balanced (50 samples per class)

## Results

Both models achieve high accuracy on the iris dataset:
- Test accuracy typically > 95%
- Excellent precision, recall, and F1-scores
- Minimal misclassifications between species

## Visualizations

The script generates:
- Feature distribution histograms
- Correlation heatmap
- Confusion matrices for both models

## Conclusions

### Key Findings

1. **Both models perform excellently** on the iris dataset with >95% accuracy
2. **Logistic Regression** provides better generalization with linear decision boundaries
3. **Decision Tree** captures non-linear patterns but with comparable accuracy
4. **Feature importance:** Petal measurements (length & width) are more discriminative than sepal measurements
5. **Class balance** makes this an ideal beginner dataset for classification

### Model Comparison

| Metric | Logistic Regression | Decision Tree |
|--------|-------------------|---------------|
| Training Accuracy | High | Very High |
| Testing Accuracy | >95% | >90% |
| Generalization | Excellent | Good |
| Interpretability | Good | Excellent |
| Training Time | Fast | Fast |

## Future Improvements

1. **Advanced Algorithms**
   - Random Forest for ensemble learning
   - SVM with different kernels
   - Gradient Boosting (XGBoost, LightGBM)

2. **Optimization**
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation for robust evaluation
   - Feature selection and engineering

3. **Model Enhancement**
   - Class weight adjustment for imbalanced data
   - Regularization techniques
   - Ensemble voting classifiers

## Author

Created as part of CSE311/312 Artificial Intelligence course project

## License

This project is open source and available under the MIT License.

---

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
