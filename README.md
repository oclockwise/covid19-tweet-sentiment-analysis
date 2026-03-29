# COVID-19 Tweet Sentiment Analysis using NLP

## Overview
This project uses natural language processing (NLP) and deep learning to classify COVID-19 related tweets into positive, neutral, or negative sentiment. It focuses on handling noisy, real-world text data such as slang, emojis, and short-form content.

## Dataset
- Source: Kaggle COVID-19 Tweet Sentiment dataset  
- ~3,798 tweets (subset used)  
- Original 5 classes merged into 3: Positive, Neutral, Negative  
- Features:
  - `OriginalTweet` (text input)
  - `Sentiment` (target label)

## Model Used
- Dense Neural Network (Fully Connected Layers)
- Dropout and regularization for overfitting control  

## Key Steps
- Data cleaning and preprocessing  
- Label encoding (text → numerical labels)  
- TF-IDF vectorization (text → numerical features)  
- Train-test split (80/20, stratified)  
- Model training, tuning, and comparison of architectures  

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC  

## Results
- Best model achieved ~68% accuracy and ~0.65 F1 score  
- Performance impacted by class imbalance and noisy text  
- Deeper models performed better than simpler architectures  

## Insights
- Tweets are challenging due to slang, sarcasm, and brevity  
- TF-IDF effectively captures important textual features  
- Model architecture and regularization significantly affect performance  
- Dense neural networks can still perform reasonably well on NLP tasks  

## Tech Stack
Python, pandas, numpy, scikit-learn, TensorFlow/Keras, matplotlib  

## Run
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
jupyter notebook
