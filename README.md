# COVID-19 Tweet Sentiment Analysis using NLP

## Overview
This project applies natural language processing (NLP) and machine learning techniques to classify the sentiment of COVID-19 related tweets. The goal is to predict whether a tweet expresses positive, neutral, or negative sentiment.

## Dataset
- Source: Kaggle COVID-19 Tweet Sentiment dataset  
- ~3,798 tweets (subset used)  
- Classes: Positive, Neutral, Negative  
- Features:
  - `OriginalTweet`
  - `Sentiment`

## Model Used
- Dense Neural Network  
- Dropout for regularization  

## Key Steps
- Data cleaning & preprocessing  
- Label encoding  
- TF-IDF vectorization  
- Train-test split (80/20, stratified)  
- Model training and tuning  

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC  

## Results
- Achieved ~68% accuracy and ~0.65 F1 score  
- Performance affected by class imbalance  
- Regularization improved generalization  

## Insights
- Tweets are noisy (slang, sarcasm)  
- TF-IDF is effective for text features  
- Class imbalance impacts results  

## Tech Stack
Python, pandas, numpy, scikit-learn, TensorFlow/Keras, matplotlib  

## 🚀 Run
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
jupyter notebook
