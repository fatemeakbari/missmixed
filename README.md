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

- Python ≥ 3.10
- NumPy
- Pandas
- scikit-learn
- XGBoost
- TensorFlow or Keras (for deep model imputation)
- tqdm

Dependencies will be installed automatically via pip.

### 📖 Usage

See the [example](./examples) folder for how to define:
Custom Iteration Architectures
Mixed-type pipelines
Trial-based imputation workflows

OR

Use Command-Line Interface (CLI)

```bash
 missmixed --path .\input_data.csv
```

#### 💻 MissMixed CLI Options

The following table lists **all command-line arguments** for MissMixed:

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--path` | `-p` | `str` | **required** | Path to the input data file (CSV or XLSX). |
| `--categorical-columns` | `-cat-col` | `str (list)` | `None` | Names of categorical columns (space-separated). Only one of the column options can be used. If none are provided, all columns are treated as continuous (default). |
| `--categorical-index` | `-cat-idx` | `int (list)` | `None` | Indices of categorical columns (space-separated). Only one of the column options can be used. |
| `--continuous-columns` | `-con-col` | `str (list)` | `None` | Names of continuous (non-categorical) columns (space-separated). Only one of the column options can be used. |
| `--continuous-index` | `-con-idx` | `int (list)` | `None` | Indices of continuous (non-categorical) columns (space-separated). Only one of the column options can be used. |
| `--initial-strategy` | `-s` | `str` | `mean` | Initial strategy for filling NaN values. Choices: `mean`, `median`, `most_frequent`. |
| `--metric` | `-m` | `str` | `r2_accuracy` | Metric for model evaluation. Choices: `r2_accuracy`, `mse`. |
| `--trials` | `-t` | `int` | `1` | Number of trials for training imputers through all iterations. |
| `--train-size` | `-ts` | `float` | `0.9` | Train size ratio (validation size = `1 - train_size`). |
| `--verbose` | `-v` | `int` | `0` | Verbosity level: `0` (silent), `1` (default), `2` (detailed). |
| `--output` | `-o` | `str` | `imputed_data.csv` | Path to save the imputed output file. |

## 📄 License

MIT License

### 📣 Citation

[1] M. M. Kalhori, M. Izadi, “A Novel Mixed-Method Approach to Missing Value Imputation: An Introduction to MissMixed”, 29th International Computer Conference, Computer Society of Iran (CSICC) – IEEE, 2025.

[2] M. M. Kalhori, M. Izadi, F. Akbari “MissMixed: An Adaptive, Extensible and Configurable Multi-Layer Framework for Iterative Missing Value Imputation”, IEEE Access, 2025 (under review).
