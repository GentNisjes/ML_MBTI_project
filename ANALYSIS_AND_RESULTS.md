# MBTI Personality Classification - Analysis & Results Documentation

## Executive Summary

This project classifies MBTI (Myers-Briggs Type Indicator) personality types from social media text data. We developed a machine learning pipeline that addresses a critical challenge: **class imbalance**. The dataset has significantly more of certain personality types (e.g., Introverts) than others (e.g., Extroverts), which causes models to perform poorly on minority classes. We explored upsampling as a solution and compared 6 different model variants.

---

## 1. The Dataset Problem: Class Imbalance

### What is Class Imbalance?

Class imbalance occurs when the training data has **unequal representation** of different classes. In the MBTI dataset, this manifests as:

- **Introversion (I) vs Extroversion (E)**: ~75% Introverts, ~25% Extroverts
- **Intuition (N) vs Sensing (S)**: ~70% Intuition, ~30% Sensing
- **Thinking (T) vs Feeling (F)**: ~55% Thinking, ~45% Feeling
- **Judging (J) vs Perception (P)**: ~50/50 balance

**Example**: If 75% of users are Introverts, a "naive" model could achieve 75% accuracy by simply predicting everyone is an Introvert—without learning anything useful about what distinguishes introverts from extroverts.

### Why This is a Problem

1. **Accuracy Metric Becomes Misleading**
   - High accuracy doesn't guarantee good performance on minority classes
   - A model predicting "all Extroverts" on a 25% Extrovert dataset gets 75% accuracy—but learns nothing

2. **Biased Predictions**
   - Models learn to heavily favor the majority class
   - Minority class predictions become extremely rare
   - Real-world harm: If detecting rare personality traits is important, the model fails

3. **Poor Minority Class Metrics**
   - **Recall for Extroverts**: Model finds few actual extroverts (low recall)
   - **Precision for Extroverts**: When it does predict extravert, it's often wrong (low precision)
   - **F1-Score**: Reflects this imbalance—accuracy alone hides the problem

### Dataset Statistics

```
Total Users: 8,675
Total Posts: ~700,000
Average Posts per User: ~81
Feature Dimension: 5,000 (TF-IDF features)

Personality Distribution (Imbalanced):
  I/E Dimension: ~3:1 ratio (Intro:Extro)
  N/S Dimension: ~2.3:1 ratio (Intuition:Sensing)
  T/F Dimension: ~1.2:1 ratio (Thinking:Feeling)
  J/P Dimension: ~1:1 ratio (Judging:Perception)
```

---

## 2. The Solution: Upsampling the Minority Class

### What is Upsampling?

Upsampling (also called **oversampling**) artificially increases the number of minority class samples to balance the training data:

```
BEFORE Upsampling:          AFTER Upsampling:
Introvert:  3,600           Introvert:  3,600  ← unchanged
Extrovert:  1,200           Extrovert:  3,600  ← duplicated to match
────────────────            ────────────────
Total:      4,800           Total:      7,200
```

**Key Points:**
- Only applied to **training data** (not test data)
- Test set remains original distribution (realistic evaluation)
- Duplicates minority class samples to equal majority class count
- Uses `RandomOverSampler` from imbalanced-learn library

### How It Helps

1. **Balances the Training Data**
   - Model sees equal representation of both classes
   - No incentive to predict only majority class
   - Learns distinctive features of both classes equally

2. **Improves Minority Class Metrics**
   - Better recall: Finds more actual minority class instances
   - Better precision: Fewer false positives for minority class
   - **F1-Score improves**: Balanced measure of classifier quality

3. **Realistic Test Evaluation**
   - Test set maintains original imbalanced distribution
   - Evaluates how model generalizes to real-world imbalance
   - Results show real-world performance, not "best-case" scenario

### When NOT to Use Upsampling

⚠️ **Trade-offs to consider:**
- Can cause **slight overfitting** (training on duplicates)
- May reduce model performance on majority class
- Doesn't add new information (just repeats existing data)
- Better alternatives: class weights, downsampling, or SMOTE

---

## 3. Data Processing Pipeline

### Step 1: Data Loading
- **Dataset**: MBTI Kaggle Dataset (8,675 users, ~700,000 posts)
- **Posts Format**: Posts separated by "|||" delimiter
- **Labels**: 16 MBTI types (4 binary dimensions)

### Step 2: Text Preprocessing
Applied comprehensive text cleaning:
- Remove URLs and web links
- Preserve sentence-ending punctuation (!, ?, .)
- Remove non-alphabetic characters
- Convert to lowercase
- Remove repeated characters (e.g., "hellooooo" → "hello")
- Remove very short (<4 chars) and very long (>30 chars) words
- **Critical**: Remove MBTI personality words from text
  - Removes: "INFP", "INTJ", "extroverted", "thinking", etc.
  - Reason: Prevents models from "cheating" by pattern-matching personality keywords
  - Ensures genuine learning of linguistic patterns, not explicit personality mentions

### Step 3: Feature Extraction (TF-IDF Vectorization)
- **TF-IDF**: Term Frequency-Inverse Document Frequency
  - Measures importance of words across the corpus
  - Common words get low weight, unique words get high weight
- **Configuration**: 5,000 most important features
- **Output**: 8,675 samples × 5,000 sparse feature matrix

### Step 4: Binary Dimension Encoding
Instead of 16-way classification, used **4 independent binary classifiers**:
- **I/E**: Introvert (1) vs Extrovert (0)
- **N/S**: Intuitive (1) vs Sensing (0)
- **T/F**: Thinking (1) vs Feeling (0)
- **J/P**: Judging (1) vs Perceiving (0)

**Advantage**: Each binary classification is simpler, more interpretable, and easier to optimize.

### Step 5: Train/Test Split
- **Split Ratio**: 80% training, 20% testing
- **Stratification**: Preserves class balance in both train and test sets
- **Random State**: Fixed seed (42) for reproducibility

---

## 4. Model Development & Training

### 4.1 Logistic Regression (Custom Implementation)

**Approach**: Gradient descent optimization from scratch

```
Algorithm: Binary Logistic Regression
  - Cost Function: Cross-entropy loss
  - Optimization: Batch gradient descent
  - Learning Rate: α = 0.0001 (very small to avoid divergence)
  - Iterations: 2,000
  - Features: 5,000 (sparse matrix)
```

**Two Variants:**
1. **Original**: Trained on imbalanced data
2. **Upsampled**: Trained on balanced data (minority class duplicated)

### 4.2 Support Vector Machine (SVM)

**Approach**: LinearSVC (linear kernel) for high-dimensional data

```
Algorithm: Support Vector Machine (Linear)
  - Kernel: Linear (best for text data)
  - Loss: Squared hinge loss
  - Solver: Dual formulation
  - Max Iterations: 5,000
```

**Two Variants:**
1. **Original**: Trained on imbalanced data
2. **Upsampled**: Trained on balanced data

### 4.3 Naive Bayes

**Approach**: Multinomial Naive Bayes with probabilistic text classification

```
Algorithm: Naive Bayes
  - Type: Multinomial (for feature counts)
  - Assumption: Features are conditionally independent
  - Feature Extraction: Both BOW and TF-IDF tested
```

**Two Variants:**
1. **Bag of Words (BOW)**: Simple word frequency counts
2. **TF-IDF**: Weighted by importance across corpus

**Note**: Naive Bayes already handles imbalance better due to probabilistic nature (no upsampled version needed).

---

## 5. Results & Interpretation

### 5.1 Model Performance Comparison (Accuracy %)

| Model | I/E | N/S | T/F | J/P | Average |
|-------|-----|-----|-----|-----|---------|
| **SVM** | 76.9 | 86.5 | 76.4 | 64.6 | **76.1** ⭐ |
| **SVM (Upsampled)** | 74.7 | 85.3 | 76.3 | 62.9 | 74.8 |
| **LR** | 77.2 | 86.1 | 53.8 | 60.2 | 69.3 |
| **LR (Upsampled)** | 73.4 | 80.7 | 75.7 | 62.1 | 73.0 |
| **NB (BOW)** | 71.6 | 78.8 | 77.9 | 66.7 | 73.7 |
| **NB (TF-IDF)** | 73.6 | 78.9 | 76.7 | 67.2 | 74.1 |

### 5.2 Best Model per Dimension

- **I/E (Introvert/Extrovert)**: **LR (77.2%)**
  - Most balanced dimension (good baseline for classification)
  
- **N/S (Intuition/Sensing)**: **SVM (86.5%)**
  - Easiest dimension (best performance across all models)
  - Strong linguistic differences between N and S traits
  
- **T/F (Thinking/Feeling)**: **NB BOW (77.9%)**
  - Naive Bayes excels here (good at capturing word preferences)
  - Strong emotional vs logical language patterns
  
- **J/P (Judging/Perception)**: **NB TF-IDF (67.2%)**
  - Hardest dimension (all models struggle)
  - Subtle differences between judging and perceiving traits

### 5.3 Key Finding: Upsampling Trade-offs

**Upsampling Results:**

| Dimension | Original | Upsampled | Change |
|-----------|----------|-----------|--------|
| **SVM I/E** | 76.9% | 74.7% | **-2.2%** |
| **SVM N/S** | 86.5% | 85.3% | **-1.2%** |
| **SVM T/F** | 76.4% | 76.3% | -0.1% |
| **SVM J/P** | 64.6% | 62.9% | **-1.7%** |
| **LR I/E** | 77.2% | 73.4% | **-3.8%** |
| **LR N/S** | 86.1% | 80.7% | **-5.4%** |
| **LR T/F** | 53.8% | 75.7% | **+21.9%** ✅ |
| **LR J/P** | 60.2% | 62.1% | **+1.9%** |

**Interpretation:**

1. **Where Upsampling Helps (T/F for LR)**
   - **LR T/F**: 53.8% → 75.7% (+21.9%!)
   - This massive improvement shows the original model was severely biased
   - On imbalanced data, LR predicted nearly everything as one class
   - Upsampling forces the model to learn both classes properly
   - **F1-Score improves dramatically**: 0.00% → 75.1%

2. **Where Upsampling Hurts (N/S for SVM)**
   - **SVM N/S**: 86.5% → 85.3% (-1.2%)
   - This dimension has good balance (2.3:1 ratio)
   - Upsampling adds redundant duplicates
   - Model memorizes duplicates slightly (overfitting)
   - But impact is small (only -1.2%)

3. **Trade-off**: 
   - Upsampling helps when initial imbalance is severe (like T/F)
   - Upsampling hurts when imbalance is moderate (like N/S)
   - Overall: **Not always beneficial** (case-dependent)

### 5.4 Dimension Difficulty Ranking

```
EASIEST  → HARDEST

1. N/S (Intuition/Sensing)    - 86.5% (SVM)
   ↓ Strong linguistic differences
   
2. T/F (Thinking/Feeling)    - 77.9% (NB BOW)
   ↓ Clear emotional vs logical patterns
   
3. I/E (Introvert/Extrovert) - 77.2% (LR)
   ↓ Moderate linguistic differences
   
4. J/P (Judging/Perception)  - 67.2% (NB TF-IDF)
   ↑ Very subtle differences
```

**Why J/P is hardest:**
- Judging ≠ "being judgmental"; Perception ≠ "perceiving"
- Subtle personality traits without obvious linguistic markers
- J/P distinction is more about life organization preferences
- Harder to detect from text alone

---

## 6. Advanced Metrics: Beyond Accuracy

### 6.1 F1-Score (Most Important for Imbalanced Data)

F1-Score balances **Precision** (of predicted positives) and **Recall** (of actual positives):

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why F1-Score matters:**
- Accuracy is misleading with class imbalance
- F1-Score penalizes models that ignore minority class
- More meaningful for imbalanced datasets

**Example (Original LR on T/F):**
- Accuracy: 53.8% (misleading—model predicts majority class)
- F1-Score: 0.00% (reveals the real problem—model learned nothing about minority class!)
- After Upsampling: F1-Score jumps to 75.1% ✅

### 6.2 Precision & Recall

**Precision**: "Of the samples I predicted as positive, how many were correct?"
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall**: "Of all actual positive samples, how many did I find?"
```
Recall = True Positives / (True Positives + False Negatives)
```

**Example Scenario (Extrovert Detection):**
- If model predicts 10 extroverts but only 3 are correct: Precision = 30%
- If there are 100 actual extroverts but model finds only 30: Recall = 30%
- F1-Score combines both concerns: both are important!

### 6.3 Overfitting Gap (Train vs Test Accuracy)

$$\text{Overfitting Gap} = \text{Train Accuracy} - \text{Test Accuracy}$$

**Interpretation:**
- **Gap < 5%**: Good generalization ✅
- **Gap 5-10%**: Acceptable, slight overfitting ⚠️
- **Gap > 10%**: Significant overfitting ❌ Model memorized training data

**Example (SVM):**
- Train Accuracy: 81.2%
- Test Accuracy: 76.9%
- Gap: +4.3% (Acceptable—model generalizes well)

---

## 7. Why This Analysis Matters

### Real-World Implications

1. **Personality Screening for Hiring**
   - If model ignores introverts (minority class), hiring discrimination
   - F1-Score reveals this; accuracy alone doesn't

2. **User Personalization**
   - Recommender systems need accurate minority class predictions
   - Accuracy alone masks poor performance on certain personality types

3. **Research Studies**
   - Publishing results with only accuracy is misleading
   - Proper metrics (F1, precision, recall) show true model quality

### Lessons Learned

✅ **Do:**
- Always check class distribution before training
- Use F1-Score, not just accuracy, for imbalanced datasets
- Test both with and without upsampling to see trade-offs
- Use stratified train/test split to preserve distribution

❌ **Don't:**
- Use accuracy as the sole metric
- Assume upsampling always improves performance
- Ignore minority class performance
- Forget to remove target-correlated features (like "INFP" from text)

---

## 8. Technical Implementation Details

### Tools & Libraries

```python
# Data Processing
pandas, numpy, scikit-learn

# Text Processing
TfidfVectorizer, CountVectorizer (sklearn)
PorterStemmer, WordNetLemmatizer (NLTK)

# Models
LogisticRegression (custom + sklearn)
LinearSVC (sklearn)
MultinomialNB (sklearn)

# Imbalanced Data
RandomOverSampler (imbalanced-learn)

# Evaluation
accuracy_score, f1_score, precision_score, recall_score
confusion_matrix, cross_val_score (sklearn)

# Persistence
joblib (model serialization)
```

### File Structure

```
notebooks/
├── mbti_preprocessing.ipynb              # Step 1: Data cleaning & vectorization
├── mbti_logistic_regression_binary_dimensions.ipynb  # Step 2a: Train LR
├── mbti_svm_binary_dimensions.ipynb      # Step 2b: Train SVM
├── naive_bayes_BOW.ipynb                 # Step 2c: Train NB (BOW)
├── naive_bayes_TF-IDF.ipynb              # Step 2d: Train NB (TF-IDF)
└── model_comparison_graphs.ipynb         # Step 3: Compare all models

models/
├── logistic_regression_binary_dimensions.pkl           # Original LR
├── logistic_regression_binary_dimensions_upsampled.pkl # Upsampled LR
├── svm_binary_dimensions.pkl                          # Original SVM
├── svm_binary_dimensions_upsampled.pkl                # Upsampled SVM
├── naive_bayes_BOW.pkl                  # NB with BOW
└── naive_bayes_TF-IDF.pkl               # NB with TF-IDF

data/processed/
├── X_vectorized.pkl                     # 5000-feature matrix
├── Y_labels.pkl                         # MBTI personality types
├── vectorizer.pkl                       # TF-IDF vectorizer
└── label_encoder.pkl                    # 16-type encoder
```

### Model Persistence Format

Each `.pkl` file contains:
```python
{
    'dimensions': ['I/E', 'N/S', 'T/F', 'J/P'],
    'models': {
        'I/E': {...},        # Model coefficients
        'N/S': {...},
        'T/F': {...},
        'J/P': {...}
    },
    'accuracies': {          # Simple accuracy
        'I/E': 77.2,
        'N/S': 86.1,
        ...
    },
    'metrics': {             # Comprehensive metrics
        'I/E': {
            'accuracy': 77.2,
            'f1_score': 76.3,
            'precision': 78.1,
            'recall': 74.6,
            'train_accuracy': 78.5,
            'overfit_gap': 1.3,
            'cv_score': 76.8
        },
        ...
    },
    'vectorizer': <TfidfVectorizer object>,
    'label_encoder': <LabelEncoder object>,
    'model_type': 'description',
    'hyperparameters': {...}
}
```

---

## 9. Conclusion

### Summary

1. **Problem**: MBTI dataset has severe class imbalance (3:1 for I/E, 2.3:1 for N/S)
2. **Impact**: Models achieve high accuracy while ignoring minority classes
3. **Solution**: Upsampling balances training data, improving minority class performance
4. **Trade-off**: Upsampling helps severely imbalanced dimensions but hurts slightly imbalanced ones
5. **Best Practice**: Use F1-Score, precision, recall—not just accuracy

### Performance Highlights

- **Best Overall Model**: SVM (76.1% average accuracy)
- **Best Balanced Performance**: SVM with consistent results across dimensions
- **Most Improved with Upsampling**: LR on T/F dimension (+21.9%)
- **Hardest to Predict**: J/P dimension (67.2% best performance)

### Recommendations

1. **For Production**: Use original SVM (no upsampling) for better balanced performance
2. **For Minority Class Focus**: Use upsampled versions when minority class detection is critical
3. **For Further Improvement**: Explore SMOTE, class weights, or ensemble methods
4. **For Research**: Always report F1-Score, precision, recall—not just accuracy

---

## 10. References & Further Reading

- **Imbalanced Learning**: [Imbalanced-Learn Documentation](https://imbalanced-learn.org/)
- **MBTI Dataset**: [Kaggle MBTI Dataset](https://www.kaggle.com/datasnaek/mbti-type)
- **Evaluation Metrics**: [Scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Class Imbalance**: [Handling Imbalanced Data](https://developers.google.com/machine-learning/crash-course/classification/handling-imbalanced-data)

---

**Document Created**: December 2024  
**Project**: MBTI Personality Classification from Text  
**Status**: Complete Analysis with 6 Model Variants
