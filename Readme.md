# Generative Data Augmentation for Life Sciences

This project demonstrates the use of generative models to augment datasets in the life sciences industry. By generating synthetic data, we can improve the performance of AI models in tasks such as drug discovery and accurate diagnosis.

## Project Overview

The project uses xthe Breast Cancer Wisconsin (Diagnostic) dataset to build a Generative Adversarial Network (GAN) that generates synthetic data. This synthetic data is then used to augment the original dataset, and a classifier is trained on the combined dataset to demonstrate improved performance.

## Features

- **Data Integration and Management**: Ingest and preprocess the Breast Cancer Wisconsin (Diagnostic) dataset.
- **Generative Model**: Build and train a GAN to generate synthetic data.
- **Data Augmentation**: Generate synthetic data and combine it with the original dataset.
- **AI Model Training**: Train a classifier on the augmented dataset.
- **Evaluation**: Evaluate the performance of the classifier on the test dataset.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shaik-abdulhaffeez/generative_data_augmentation_life_sciences.git
    cd generative-data-augmentation-life-sciences
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Prepare the dataset:
    - Load and preprocess the Breast Cancer Wisconsin (Diagnostic) dataset.

2. Build and train the GAN:
    - Define the generator and discriminator networks.
    - Train the GAN to generate synthetic data.

3. Generate synthetic data:
    - Use the trained GAN to generate synthetic samples.

4. Train the classifier:
    - Combine the synthetic data with the original dataset.
    - Train a classifier on the augmented dataset.

5. Evaluate the classifier:
    - Evaluate the performance of the classifier on the test dataset.

### Example Notebook

An example Jupyter notebook is provided to demonstrate the entire workflow. You can find the notebook in the `notebooks` directory.

### Project Structure
```
generative-data-augmentation-life-sciences/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── notebooks/              # Jupyter notebooks
│   └── data_augmentation.ipynb
├── src/                    # Source code
│   ├── data_preparation.py
│   ├── gan_model.py
│   ├── train_gan.py
│   ├── generate_synthetic_data.py
│   ├── train_classifier.py
│   └── evaluate_model.py
├── requirements.txt        # Required libraries
├── README.md               # Project README
└── .gitignore              # Git ignore file
```

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

### Acknowledgements

- The Breast Cancer Wisconsin (Diagnostic) dataset is provided by the UCI Machine Learning Repository.
- The project uses open-source libraries such as PyTorch, NumPy, and scikit-learn.