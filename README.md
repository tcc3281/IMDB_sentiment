# IMDB Sentiment Analysis with BERT

This project demonstrates sentiment analysis on the IMDB dataset using the BERT (Bidirectional Encoder Representations from Transformers) model. The implementation leverages the `transformers` library by Hugging Face for fine-tuning the pre-trained BERT model.

## Overview

The goal of this project is to classify movie reviews as either positive or negative using a fine-tuned BERT model. The pipeline includes data preprocessing, tokenization, model training, evaluation, and visualization of results.

## Key Features

- **Pre-trained BERT Model**: Utilizes `bert-base-uncased` from Hugging Face for sequence classification.
- **Data Preprocessing**: Includes HTML tag removal, URL removal, and sentiment mapping.
- **Custom Dataset Class**: Implements a PyTorch `Dataset` for handling tokenized data.
- **Fine-tuning**: Freezes BERT's base layers and fine-tunes the classification head.
- **Early Stopping**: Implements early stopping to prevent overfitting.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, and confusion matrix.

## Steps

1. **Load Data**: The IMDB dataset is loaded and preprocessed to remove noise.
2. **Preprocess Data**: Reviews are cleaned by removing HTML tags and URLs, and sentiments are mapped to binary labels.
3. **Tokenize Data**: Reviews are tokenized using the BERT tokenizer with a maximum sequence length of 512.
4. **Prepare DataLoader**: Tokenized data is wrapped in a custom PyTorch `Dataset` and loaded using `DataLoader`.
5. **Train Model**: The BERT model is fine-tuned on the training data with early stopping based on validation loss.
6. **Evaluate Model**: The model is evaluated on the test set, and metrics like accuracy, precision, recall, and F1-score are calculated.
7. **Visualize Results**: Training/validation loss and accuracy are plotted, and a confusion matrix is displayed.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- BeautifulSoup

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/tcc3281/IMDB_sentiment.git
   cd IMDB_sentiment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook bert_model.ipynb
   ```

## Results

- **Accuracy**: Achieved high accuracy on the test set.
- **Confusion Matrix**: Visualized the model's performance in predicting positive and negative sentiments.
- **Training/Validation Loss**: Monitored during training to ensure convergence.

## References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

