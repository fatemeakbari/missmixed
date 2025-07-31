# MissMixed

## A Configurable Framework for Iterative Missing Data Imputation

**MissMixed** is a Python library designed for flexible and modular imputation of missing values in tabular datasets. It supports a wide range of imputation strategies, including ensemble methods, trial-based model selection, and deep learning integration — all within a customizable iterative architecture.

## 🔍 What is MissMixed?

MissMixed is not just a single algorithm — it’s a **framework** for building **iteration-wise, model-aware imputation pipelines**. It enables users to:

- Handle continuous, categorical, or mixed-type features
- Define custom model configurations at each iteration
- Combine multiple imputation algorithms (e.g., RandomForest, KNN, Deep Neural Networks)
- Dynamically evaluate and update imputed values using internal validation

Whether you’re working with low-dimensional medical data or large-scale mixed-type datasets, MissMixed is designed to offer **accuracy**, **adaptability**, and **interpretability**.

## 🚀 Installation

```bash
pip install missmixed
```

### 📦 Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- scikit-learn
- XGBoost
- TensorFlow or Keras (for deep model imputation)
- tqdm

Dependencies will be installed automatically via pip.

### 📖 Usage

See the examples folder for how to define:
Custom Iteration Architectures
Mixed-type pipelines
Trial-based imputation workflows

### 📄 License

MIT License


### 📣 Citation

[1] M. M. Kalhori, M. Izadi, “A Novel Mixed-Method Approach to Missing Value Imputation: An Introduction to MissMixed”, 29th International Computer Conference, Computer Society of Iran (CSICC) – IEEE, 2025.

[2] M. M. Kalhori, F. Akbari, M. Izadi, “MissMixed: An Adaptive and Configurable Framework for Iterative Missing Value Imputation”, IEEE Access, 2025 (under review).
