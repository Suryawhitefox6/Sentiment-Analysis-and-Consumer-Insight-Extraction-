# Sentiment Analysis of Amazon Product Reviews

This repository contains the code, data, and documentation for a comprehensive study on sentiment analysis of Amazon product reviews using deep learning models. The research focuses on two product categories: Health & Personal Care and Handmade Products. The goal is to develop a sophisticated sentiment analysis model that incorporates textual, metadata, and behavioral signals to enhance the accuracy and insights of customer sentiment classification.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)
9. [References](#references)
10. [Setup Instructions](#setup-instructions)
11. [Usage](#usage)
12. [Contributing](#contributing)
13. [License](#license)

---

## Introduction

The growth of e-commerce has made online customer reviews a critical factor in consumer decision-making. Sentiment analysis of these reviews provides valuable insights into customer preferences, opinions, and satisfaction. This study leverages advanced deep learning models, including DistilBERT, LSTM, TextCNN, and hybrid architectures, to analyze sentiment in Amazon product reviews. By incorporating metadata and behavioral signals, the research aims to deliver more accurate and holistic sentiment insights.

---

## Dataset

The dataset used in this study is the **Amazon Reviews 2023 dataset** from McAuley Lab, which includes:
- **571.54 million reviews** spanning from May 1996 to September 2023.
- Two categories: **Health & Personal Care** and **Handmade Products**.
- Rich textual and metadata information, such as review text, ratings, helpfulness votes, product descriptions, prices, and user-item interactions.

### Dataset Link
You can access the dataset [here](https://amazon-reviews-2023.github.io/).

### Dataset Structure
1. **Metadata CSV**: Contains product information like category, title, average rating, price, and availability date.
2. **Reviews CSV**: Includes customer feedback such as ratings, review text, helpful votes, purchase verification status, and review date.

### Sample Data
- **Health & Personal Care**: 494.1K reviews, 461.7K users, 60.3K products.
- **Handmade Products**: 664.2K reviews, 586.6K users, 164.7K products.

---

## Methodology

The methodology involves the following steps:
1. **Data Preprocessing**: Handling missing values, standardizing date formats, price normalization, categorical data encoding, duplicate removal, and merging metadata with review data.
2. **Feature Engineering**: Text preprocessing, sentiment labeling, numerical and categorical feature extraction, and word embeddings (TF-IDF, Word2Vec, BERT).
3. **Model Training**: Using DistilBERT, LSTM, TextCNN, and hybrid CNN+BiLSTM architectures for sentiment classification.
4. **Model Evaluation**: Evaluating models using accuracy, precision, recall, and F1-score.

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset's structure, distributions, and trends:
- **Rating Distribution**: Most reviews are 4-star and 5-star, indicating high customer satisfaction.
- **Review Length**: Positive reviews are shorter, while negative reviews are longer and more detailed.
- **Word Clouds**: Visual representation of frequently occurring words in positive and negative reviews.
- **Sentiment Trends**: Analysis of sentiment polarity over time.
- **Correlation Matrix**: Relationships between ratings, sentiment scores, and prices.

---

## Model Training and Evaluation

### Models Used
1. **DistilBERT**: A lightweight transformer model with high accuracy and computational efficiency.
2. **LSTM**: Captures long-range dependencies in sequential data.
3. **TextCNN**: Extracts local features from text data.
4. **Hybrid CNN+BiLSTM**: Combines CNN for feature extraction and BiLSTM for sequential pattern detection.

### Evaluation Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of correctly predicted positive instances.
- **Recall**: Ability to identify all relevant positive instances.
- **F1-Score**: Harmonic mean of precision and recall.

---

## Results

### Health & Personal Care
- **DistilBERT**: 93.7% accuracy.
- **LSTM**: 92% accuracy.
- **TextCNN**: 92.18% accuracy.
- **Hybrid CNN+BiLSTM**: 92.25% accuracy.

### Handmade Products
- **DistilBERT**: 96.36% accuracy.
- **LSTM**: 90.56% accuracy.
- **TextCNN**: 95.83% accuracy.
- **Hybrid CNN+BiLSTM**: 95.78% accuracy.

---

## Conclusion

DistilBERT outperformed other models in both categories, demonstrating its effectiveness in capturing sentiment nuances. The study highlights the importance of incorporating metadata and behavioral signals for more accurate sentiment analysis. Future work can focus on domain-specific fine-tuning, multimodal analysis, and real-time sentiment monitoring.

---

## Future Work
- Fine-tune transformer models using domain-specific embeddings.
- Incorporate multimodal data (images, videos, user interactions) for broader insights.
- Implement aspect-based sentiment analysis (ABSA) for more granular insights.
- Develop real-time sentiment monitoring systems.

---

## References
- [McAuley Lab Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
- [DistilBERT: A smaller, faster, cheaper, and lighter Transformer](https://arxiv.org/abs/1910.01108)

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis.git
