# MBTI Personality Type Prediction from Text

A machine learning project that predicts Myers-Briggs Type Indicator (MBTI) personality dimensions from social media text posts using natural language processing and multiple classification algorithms.

## Project Overview

This project implements and compares multiple machine learning models to predict the four binary MBTI personality dimensions:
- **I/E**: Introversion vs. Extraversion
- **N/S**: Intuition vs. Sensing
- **T/F**: Thinking vs. Feeling
- **J/P**: Judging vs. Perceiving

The models treat each dimension as an independent binary classification task, using text features extracted from user posts to make predictions.

## Dataset

The project uses the MBTI Myers-Briggs Personality Type Dataset from Kaggle, containing over 8,600 rows of social media posts labeled with personality types.

**Dataset Source**: M. J. (datasnaek), "(MBTI) Myers-Briggs Personality Type Dataset," Kaggle, 2017. [Online]. Available: https://www.kaggle.com/datasnaek/mbti-type

## Models Implemented

**Common Features Across All Models:**
- Binary classification for each of the four MBTI dimensions
- RandomOverSampler for handling class imbalance
- 5-fold stratified cross-validation for robust performance estimation
- Comprehensive evaluation metrics: accuracy, F1-score, precision, recall, overfitting gap

### Model-Specific Implementations:

#### 1. [Support Vector Machine (Linear SVC)](notebooks/mbti_svm.ipynb)
- Linear kernel with optimal hyperparameters
- Manual C parameter override (C=0.01) to reduce overfitting
- Strong regularization for improved generalization

#### 2. [Logistic Regression](notebooks/mbti_logistic_regression.ipynb)
- L1 (Lasso) and L2 (Ridge) regularization options
- [Systematic hyperparameter tuning](notebooks/logistic_regression_hyperparameter_tuning.ipynb) with GridSearchCV
- Generalization-aware parameter selection framework
- Exports three parameter configurations: standard, generalization-aware, and hand-picked L1

#### 3. Naive Bayes
- **[Bag of Words (BoW)](notebooks/naive_bayes_BOW.ipynb)**: Simple word frequency features with 5-fold CV
- **[TF-IDF](notebooks/naive_bayes_TF-IDF.ipynb)**: Weighted features emphasizing distinctive words with 5-fold CV
- Multinomial Naive Bayes with probabilistic classification

## Key Features

- **[Comprehensive Preprocessing](notebooks/mbti_preprocessing.ipynb)**: Text cleaning, aggregation, and TF-IDF vectorization
- **Overfitting Analysis**: Train vs. test performance comparison using both accuracy and F1-score metrics
- **Hyperparameter Tuning**: Systematic optimization with generalization considerations
- **[Model Comparison](notebooks/model_comparison_graphs.ipynb)**: Detailed performance metrics across all models and dimensions
- **Visualization**: Heatmaps, confusion matrices, and overfitting gap analysis

## Project Structure

```
ML_MBTI_project/
├── data/
│   └── processed/
│       └── preprocessed_data.csv
├── models/
│   ├── svm_binary_dimensions_upsampled.pkl
│   ├── logistic_regression_binary_dimensions_optimal.pkl
│   ├── naive_bayes_BOW.pkl
│   └── naive_bayes_TF-IDF.pkl
├── notebooks/
│   ├── mbti_preprocessing.ipynb
│   ├── mbti_svm.ipynb
│   ├── mbti_logistic_regression.ipynb
│   ├── logistic_regression_hyperparameter_tuning.ipynb
│   ├── naive_bayes_BOW.ipynb
│   ├── naive_bayes_TF-IDF.ipynb
│   └── model_comparison_graphs.ipynb
└── reference_papers/
```

## Results

Performance varies across dimensions, with N/S being the most challenging due to class imbalance and subtle linguistic differences. The project includes detailed analysis of:
- F1-scores and accuracy metrics
- Precision-recall trade-offs
- Overfitting gaps (train vs. test performance)
- Cross-validation scores

Detailed results and visualizations are available in [model_comparison_graphs.ipynb](notebooks/model_comparison_graphs.ipynb) and [ANALYSIS_AND_RESULTS.md](ANALYSIS_AND_RESULTS.md).

## References

[1] S. Chaudhary, R. Singh, S. T. Hasan, and I. Kaur, "A Comparative Study of Different Classifiers for Myers-Brigg Personality Prediction Model," International Research Journal of Engineering and Technology (IRJET), vol. 5, no. 5, pp. 1410–1413, May 2018.

[2] S. Patel, M. Nimje, A. Shetty, and S. Kulkarni, "Personality Analysis using Social Media," International Journal of Engineering Research & Technology (IJERT), vol. 9, no. 3 (Special Issue: NTASU-2020 Conference Proceedings), pp. 306–309, 2021.

[3] Z. Mushtaq, S. Ashraf, and N. Sabahat, "Predicting MBTI Personality type with K-means Clustering and Gradient Boosting," in 2020 IEEE 23rd International Multitopic Conference (INMIC), Bahawalpur, Pakistan, 2020, pp. 1–5, doi: 10.1109/INMIC50486.2020.9318078.

[4] M. J. (datasnaek), "(MBTI) Myers-Briggs Personality Type Dataset," Kaggle, 2017. [Online]. Available: https://www.kaggle.com/datasnaek/mbti-type

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn
- joblib

## Usage

1. Download the dataset from Kaggle and place it in the appropriate directory
2. Run [mbti_preprocessing.ipynb](notebooks/mbti_preprocessing.ipynb) to preprocess the raw data
3. Train models using the respective notebooks ([SVM](notebooks/mbti_svm.ipynb), [Logistic Regression](notebooks/mbti_logistic_regression.ipynb), [Naive Bayes](notebooks/naive_bayes_BOW.ipynb))
4. Compare results using [model_comparison_graphs.ipynb](notebooks/model_comparison_graphs.ipynb)

## License

This project is for academic purposes as part of a Machine Learning course at KU Leuven.
