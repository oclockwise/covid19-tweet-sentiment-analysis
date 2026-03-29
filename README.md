# COVID-19 Tweet Sentiment Analysis using NLP

## Overview
This project applies natural language processing (NLP) and machine learning techniques to classify the sentiment of COVID-19 related tweets. The goal is to predict whether a tweet expresses positive, neutral, or negative sentiment based on its text content.

## Dataset
- Source: Kaggle COVID-19 Tweet Sentiment dataset  
- ~3,798 tweets (subset used)  
- Original 5 classes merged into 3: Positive, Neutral, Negative :contentReference[oaicite:0]{index=0}  
- Features used:
  - `OriginalTweet` (text)
  - `Sentiment` (label)

## Model Used
- Dense Neural Network (Fully Connected Layers)
- Dropout for regularization  

Model design follows a constrained deep learning approach using only dense and dropout layers :contentReference[oaicite:1]{index=1}  

## Key Steps
- Data cleaning (removed duplicates, selected relevant columns)  
- Label encoding (converted sentiment to numerical values) :contentReference[oaicite:2]{index=2}  
- Text preprocessing and vectorization using TF-IDF  
- Feature extraction (top 10,000 words) :contentReference[oaicite:3]{index=3}  
- Train-test split (80/20, stratified) :contentReference[oaicite:4]{index=4}  
- Model training and tuning  

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC (ROC Curve) :contentReference[oaicite:5]{index=5}  

## Results
- Best model achieved ~68% accuracy and ~0.65 F1 score :contentReference[oaicite:6]{index=6}  
- Performance impacted by class imbalance (fewer neutral tweets) :contentReference[oaicite:7]{index=7}  
- Model effectiveness improved with dropout and regularization  

## Insights
- NLP tasks are challenging due to slang, sarcasm, and short text  
- TF-IDF effectively captures important words in tweets  
- Class imbalance affects model performance, especially for neutral class  
- Simpler dense models can still perform reasonably well on real-world data  

## Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  
- TensorFlow / Keras  
- matplotlib  

## Run
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
jupyter notebook
