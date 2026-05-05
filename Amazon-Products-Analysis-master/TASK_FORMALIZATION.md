# Task Formalization: Hybrid Recommender System
## Amazon Home & Kitchen Q&A Dataset

---

## 📊 THE CORE PROBLEM

**Task:** Rating Prediction (Regression Problem)

**Goal:** Predict a user's rating for a product (item) based on collaborative filtering patterns and content features.

---

## 🎯 THE TARGET VARIABLE (Y)

### What is Y?

**Y = `rating`** (a numerical value from 1 to 5)

### Where does it come from?

Since the Amazon Q&A dataset **does NOT** have explicit ratings, we **synthetically create** ratings from sentiment analysis:

```python
# From sentiment_cf_analysis.ipynb, Cell around line 816:

def sentiment_to_rating(compound):
    """Convert sentiment compound score (-1 to 1) to rating (1 to 5)."""
    # VADER sentiment compound ranges from -1 (very negative) to +1 (very positive)
    # We scale it to rating range [1, 5]
    return ((compound + 1) / 2) * 4 + 1

df['rating'] = df['sentiment_compound'].apply(sentiment_to_rating)
```

**Explanation:**
- VADER sentiment analysis gives us a `sentiment_compound` score from -1 to +1
- Compound = -1 (very negative answer) → Rating = 1
- Compound = 0 (neutral answer) → Rating = 3
- Compound = +1 (very positive answer) → Rating = 5

**Why use sentiment as rating?**
- The original dataset has Questions & Answers but no explicit ratings
- Positive/helpful answers indicate user satisfaction (high rating)
- Negative/unhelpful answers indicate dissatisfaction (low rating)
- This is a **proxy** for user preference/satisfaction

---

## 👤 WHO IS THE "USER"?

**User = The person who wrote the ANSWER** (answerer)

Since the dataset doesn't have explicit user IDs, we create synthetic user IDs:

```python
def create_user_id(text):
    """Create user ID by hashing answer text."""
    if pd.isna(text):
        return 'unknown'
    # Hash the answer text to create a pseudo-user ID
    return hashlib.md5(str(text).encode()).hexdigest()[:8]

df['user_id'] = df['answer'].apply(create_user_id)
```

**Assumption:** Same answer text = same user (approximation)

---

## 🔢 THE MACHINE LEARNING TASK

### Problem Type
**Supervised Regression**

### Training Data Structure

Each training example is a **(user, item, features) → rating** tuple:

```
X_train (Features):
  - CF Features (10 features)
  - Content Features (19 features)
  Total: 29 features

y_train (Target):
  - rating (1 to 5, continuous)
```

### What we're predicting

Given a user-item pair and associated features, predict:

> **"What rating would this user give to this item?"**

---

## 🧩 FEATURE GROUPS

### 1️⃣ **CF (Collaborative Filtering) Features** [10 features]

These capture user-item interaction patterns:

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `user_mean_rating` | Average rating this user gives | 3.8 |
| `item_mean_rating` | Average rating this item receives | 4.2 |
| `user_item_count` | How many items this user rated | 15 |
| `item_user_count` | How many users rated this item | 42 |
| `max_item_similarity` | Max similarity to items this user rated | 0.85 |
| `mean_item_similarity` | Avg similarity to items this user rated | 0.42 |
| `max_user_similarity` | Max similarity to users who rated this item | 0.73 |
| `mean_user_similarity` | Avg similarity to users who rated this item | 0.35 |
| `item_cf_prediction` | Item-Item CF rating prediction | 4.1 |
| `user_cf_prediction` | User-User CF rating prediction | 3.9 |

**Purpose:** Capture collaborative patterns ("users like you also liked...")

---

### 2️⃣ **Content Features** [19 features]

These capture properties of the question-answer text:

#### A. Sentiment Features (4)
| Feature | Description | Range |
|---------|-------------|-------|
| `sentiment_neg` | Negative sentiment score | 0 to 1 |
| `sentiment_neu` | Neutral sentiment score | 0 to 1 |
| `sentiment_pos` | Positive sentiment score | 0 to 1 |
| `sentiment_compound` | Overall sentiment | -1 to 1 |

#### B. Readability Features (4)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `flesch_reading_ease` | How easy to read (0-100) | Higher = easier |
| `flesch_kincaid_grade` | Grade level needed | ~8 = 8th grade |
| `gunning_fog` | Complexity score | Higher = more complex |
| `smog_index` | Readability index | Higher = harder |

#### C. Lexical Features (11)
| Feature | Description |
|---------|-------------|
| `exclamation_count` | Number of `!` in answer |
| `question_mark_count` | Number of `?` in answer |
| `period_count` | Number of `.` in answer |
| `comma_count` | Number of `,` in answer |
| `total_punctuation` | Total punctuation marks |
| `sentence_count` | Number of sentences |
| `avg_word_length` | Average word length |
| `question_word_count` | Words in question |
| `answer_word_count` | Words in answer |
| `question_char_len` | Characters in question |
| `answer_char_len` | Characters in answer |

**Purpose:** Capture content quality signals (detailed, well-written answers may indicate higher satisfaction)

---

## 📈 THE WORKFLOW

### Stage 1: `sentiment_cf_analysis.ipynb`

1. **Load raw Q&A data**
2. **Generate synthetic ratings** from sentiment
3. **Create user IDs** from answer text hashing
4. **Extract content features**:
   - Sentiment analysis (VADER)
   - Readability scores (textstat)
   - Lexical features (custom extraction)
5. **Build user-item matrix**
6. **Implement pure CF models** (Item-Item, User-User)
7. **Evaluate CF models** with RMSE/MAE
8. **Save enhanced dataset** (`enhanced_home_kitchen_qa.pkl`)

### Stage 2: `hybrid_recommender.ipynb`

1. **Load enhanced dataset** (with ratings + features)
2. **Rebuild CF infrastructure** (user-item matrix, similarities)
3. **Extract CF-based features** (similarities, predictions)
4. **Extract content features** (sentiment, readability, lexical)
5. **Concatenate all features** → X_hybrid (29 features)
6. **Define target** → y = rating (1-5)
7. **Train ML models**:
   - Linear Regression (CF only, Content only, Hybrid)
   - Random Forest (CF only, Content only, Hybrid)
8. **Compare models** to pure CF baseline
9. **Analyze feature importance**

---

## 🎓 THE SCIENTIFIC QUESTION

**"Can we improve rating prediction by combining collaborative filtering signals with content-based features?"**

### Hypothesis

Pure CF models use only user-item interaction patterns. By adding content features (sentiment, readability, text quality), we can:

1. **Cold Start Mitigation:** Predict ratings for new users/items using content
2. **Richer Signals:** Capture aspects CF misses (answer quality, sentiment)
3. **Better Accuracy:** Combine both worlds for improved predictions

### Evaluation

Compare models using:
- **RMSE** (Root Mean Square Error) - penalizes large errors
- **MAE** (Mean Absolute Error) - average prediction error
- **R²** (Coefficient of Determination) - variance explained

---

## 📊 EXAMPLE: ONE TRAINING INSTANCE

```python
# User-Item Pair
user_id = "a1b2c3d4"  # (hash of answer text)
item_id = "B00123XYZ"  # (product ASIN)

# Features (X)
CF Features:
  - user_mean_rating: 3.8
  - item_mean_rating: 4.2
  - user_item_count: 12
  - item_user_count: 45
  - max_item_similarity: 0.82
  - mean_item_similarity: 0.45
  - max_user_similarity: 0.68
  - mean_user_similarity: 0.34
  - item_cf_prediction: 4.1
  - user_cf_prediction: 3.9

Content Features:
  - sentiment_compound: 0.72 (positive)
  - flesch_reading_ease: 65.4 (moderate)
  - answer_word_count: 85
  - ... (16 more features)

# Target (y)
rating = 4.36  
# (derived from sentiment_compound = 0.72)
# Formula: ((0.72 + 1) / 2) * 4 + 1 = 4.36
```

### Model Training

```python
# Linear Regression learns:
y_pred = β₀ + β₁(user_mean_rating) + β₂(item_mean_rating) + ... + β₂₉(answer_char_len)

# Random Forest learns:
y_pred = f(all_features)  # non-linear decision trees
```

### Prediction at Test Time

Given a new (user, item) pair:
1. Extract 10 CF features
2. Extract 19 content features
3. Feed to trained model
4. Output: Predicted rating (e.g., 4.2)
5. Compare to actual rating (e.g., 4.5)
6. Calculate error: |4.5 - 4.2| = 0.3

---

## 🔑 KEY INSIGHTS

### Why is this a valid approach?

1. **Sentiment as Proxy:** Positive answers indicate user satisfaction
2. **Content Matters:** Well-written, detailed answers reflect quality
3. **Hybrid Power:** CF patterns + content features = comprehensive model
4. **Real-world Applicable:** Many Q&A/forum datasets lack explicit ratings

### What are we learning?

The model learns patterns like:
- "Users who write positive, detailed answers tend to give higher ratings"
- "Items with consistently positive answers are rated higher"
- "Similar users have similar rating patterns"
- "Answer quality (readability, length) correlates with satisfaction"

### Business Value

This approach can:
- **Predict user satisfaction** from Q&A text
- **Identify high-quality products** (consistent positive answers)
- **Recommend products** based on Q&A patterns
- **Surface helpful answers** (those indicating high satisfaction)

---

## 📝 SUMMARY TABLE

| Component | Value/Description |
|-----------|-------------------|
| **Task Type** | Supervised Regression |
| **Target (Y)** | `rating` (1-5, continuous) |
| **Y Source** | Sentiment analysis (VADER compound score scaled to 1-5) |
| **User** | Answer author (synthetic ID from text hash) |
| **Item** | Product (ASIN) |
| **Features (X)** | 29 features (10 CF + 19 Content) |
| **Training Samples** | 22,575 user-item-rating tuples |
| **Test Samples** | 5,644 user-item-rating tuples |
| **Evaluation Metrics** | RMSE, MAE, R² |
| **Baseline** | Global mean rating |
| **Models Compared** | 9 models (Baseline, Pure CF, LR variants, RF variants) |

---

## 🎯 THE ANSWER TO YOUR QUESTION

**Q: "What is our Y variable?"**

**A:** `rating` (1-5) - a synthetic rating derived from VADER sentiment analysis of answer text, where positive sentiment = high rating, negative sentiment = low rating.

**Q: "What exactly are we doing on the features?"**

**A:** We're building a **hybrid rating prediction system** that:
1. Uses 10 CF features to capture collaborative patterns (user-user, item-item similarities)
2. Uses 19 content features to capture text quality (sentiment, readability, lexical properties)
3. Trains ML models (Linear Regression, Random Forest) to predict ratings
4. Compares hybrid models to pure CF to see if content features improve accuracy

**The core hypothesis:** Combining what other users like (CF) with what the content says (sentiment/quality) gives better predictions than CF alone.

---

Generated: December 2025
Dataset: Amazon Home & Kitchen Q&A




