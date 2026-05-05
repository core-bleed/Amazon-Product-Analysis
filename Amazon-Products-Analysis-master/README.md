# Amazon Q&A Recommender System Project
## Home and Kitchen Category Analysis

---

## 📁 Project Structure

```
Project/
│
├── 📓 notebooks/                          # Jupyter notebooks
│   ├── eda_amazon_qa.ipynb               # Exploratory Data Analysis
│   ├── sentiment_cf_analysis.ipynb       # Sentiment Analysis & Collaborative Filtering
│   ├── hybrid_recommender.ipynb          # Hybrid Recommender System
│   └── svd_matrix_factorization.ipynb    # Matrix Factorization with SVD
│
├── 📊 visualizations/                     # All visualizations organized by analysis type
│   ├── eda/                              # EDA visualizations
│   │   ├── question_type_distribution.png
│   │   ├── answer_type_distribution.png
│   │   ├── asin_distribution.png
│   │   ├── text_length_distributions.png
│   │   ├── text_length_boxplots.png
│   │   ├── text_length_by_type.png
│   │   ├── question_vs_answer_length.png
│   │   └── temporal_analysis.png
│   │
│   ├── sentiment/                        # Sentiment analysis visualizations
│   │   ├── sentiment_analysis.png
│   │   └── sentiment_by_answer_type.png
│   │
│   └── cf/                               # Collaborative filtering visualizations
│       ├── readability_analysis.png
│       ├── lexical_features.png
│       ├── cf_evaluation.png
│       ├── prediction_scatter.png
│       └── error_distribution.png
│   │
│   ├── hybrid/                           # Hybrid recommender visualizations
│   │   ├── model_comparison.png
│   │   ├── feature_importance.png
│   │   └── prediction_comparison.png
│   │
│   └── svd/                              # Matrix factorization visualizations
│       ├── svd_cf_comparison.png
│       └── prediction_scatter_plots.png
│
├── 💾 data/                               # Data directory
│   ├── raw/                              # Raw data files
│   │   └── qa_Home_and_Kitchen.json.gz
│   │
│   └── processed/                        # Processed data files
│       ├── processed_home_kitchen_qa.pkl   # After EDA
│       └── enhanced_home_kitchen_qa.pkl    # After sentiment & CF
│
├── 🐍 scripts/                            # Python utility scripts
│   ├── download_amazon_qa_data.py        # Download Amazon Q&A dataset
│   └── data_utils.py                     # Data processing utilities
│
├── 📈 results/                            # Analysis results and reports
│   ├── cf_evaluation_results.csv         # CF model evaluation metrics
│   ├── hybrid_model_comparison.csv       # Hybrid model comparison results
│   └── svd_cf_comparison.csv             # SVD vs CF comparison results
│
└── 📄 README.md                           # This file

```

---

## 📋 Notebooks Description

### 1. `eda_amazon_qa.ipynb` - Exploratory Data Analysis
**Sections:**
- Setup and Data Loading (using `parse()` and `getDF()` functions)
- Basic Data Overview (shape, columns, missing values, duplicates)
- Question Type Distribution Analysis (yes/no vs open-ended)
- Product (ASIN) Distribution Analysis
- Text Analysis (question & answer length)
- Temporal Analysis (trends over time)
- Sample Questions and Answers
- Summary Statistics

**Output:**
- `data/processed/processed_home_kitchen_qa.pkl`
- Visualizations in `visualizations/eda/`

---

### 2. `sentiment_cf_analysis.ipynb` - Sentiment Analysis & Collaborative Filtering
**Sections:**
- Setup and Data Loading
- Sentiment Analysis on Answers (VADER)
  - Compound scores (neg, neu, pos, compound)
  - Sentiment classification (Positive/Neutral/Negative)
- Readability Scores
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - Automated Readability Index
  - Coleman-Liau Index
- Lexical Features (Punctuation Analysis)
  - Exclamation marks, question marks, periods, commas
  - Total punctuation, sentence count, avg word length
- User-Item Matrix Construction
  - User: Hash of answer text (proxy for answerer)
  - Item: Product ASIN
  - Rating: Normalized sentiment (1-5 scale)
- Collaborative Filtering Implementation
  - Item-Item CF (k=20, cosine similarity)
  - User-User CF (k=20, cosine similarity, mean-centered)
- Model Evaluation (RMSE/MAE)
- Summary and Results

**Output:**
- `data/processed/enhanced_home_kitchen_qa.pkl`
- `results/cf_evaluation_results.csv`
- Visualizations in `visualizations/sentiment/` and `visualizations/cf/`

---

### 3. `hybrid_recommender.ipynb` - Hybrid Recommender System
**Sections:**
- Combining Collaborative Filtering with Content Features
- CF-based features extraction (user/item similarities, predictions)
- Content features integration (sentiment, readability, lexical)
- Machine Learning models (Linear Regression, Random Forest)
- Comparison: Pure CF vs Hybrid Models
- Feature importance analysis
- Model evaluation and visualization

**Output:**
- `data/results/hybrid_model_comparison.csv`
- Visualizations in `visualizations/hybrid/`

---

### 4. `svd_matrix_factorization.ipynb` - Matrix Factorization with SVD
**Sections:**
- Custom SVD implementation using Stochastic Gradient Descent
- TruncatedSVD and NMF implementations
- Comparison with Traditional Collaborative Filtering
- Latent factor extraction for hybrid models
- Performance evaluation and visualization

**Output:**
- `data/results/svd_cf_comparison.csv`
- `data/results/svd_train_features.pkl` and `svd_test_features.pkl`
- `data/results/svd_latent_factors.pkl`
- Visualizations in `visualizations/svd/`

---

## 👥 Team & Work Division

### Sachin
- **Exploratory Data Analysis (EDA)**: Complete data exploration, analysis, and visualization in `eda_amazon_qa.ipynb`
- **Hybrid Recommender System**: Partial contribution to `hybrid_recommender.ipynb` implementation

### Muhammad Asad
- **Collaborative Filtering (CF)**: Implementation of Item-Item and User-User collaborative filtering algorithms in `sentiment_cf_analysis.ipynb`
- CF model evaluation and performance analysis

### Amir Raj
- **Matrix Factorization (SVD)**: Complete implementation of SVD using Stochastic Gradient Descent in `svd_matrix_factorization.ipynb`
- Comparison of SVD with traditional CF methods
- Latent factor extraction and feature engineering

---

## 🔮 Future Work

- **Deep Learning Approaches**: Explore neural collaborative filtering and deep learning-based recommendation models to capture complex non-linear patterns in user-item interactions.
- **Real-time Recommendation System**: Develop a production-ready recommendation API that can handle real-time queries and update recommendations dynamically based on user behavior patterns.
- **Multi-objective Optimization**: Investigate recommendation strategies that balance multiple objectives such as relevance, diversity, novelty, and serendipity for enhanced user satisfaction.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install vaderSentiment textstat
```

### Data Download
```bash
python scripts/download_amazon_qa_data.py
```

### Run Notebooks
1. Open `notebooks/eda_amazon_qa.ipynb` - Run all cells for EDA
2. Open `notebooks/sentiment_cf_analysis.ipynb` - Run all cells for sentiment & CF analysis
3. Open `notebooks/hybrid_recommender.ipynb` - Run all cells for hybrid recommender system
4. Open `notebooks/svd_matrix_factorization.ipynb` - Run all cells for SVD matrix factorization

---

## 📊 Dataset Information

**Source:** Julian McAuley, UCSD  
**Category:** Home and Kitchen  
**Total Questions:** 184,439  
**Total Products (ASINs):** ~67,000  

**Data Fields:**
- `asin` - Product ID
- `questionType` - 'yes/no' or 'open-ended'
- `answerType` - 'Y', 'N', or '?' (for yes/no questions)
- `answerTime` - Raw answer timestamp
- `unixTime` - Unix timestamp
- `question` - Question text
- `answer` - Answer text

---

## 📈 Key Findings

### Sentiment Analysis
- Most answers have **positive sentiment** (helpful community)
- Mean compound score: ~0.2-0.3 (positive)
- Sentiment distribution: ~40-50% Positive, ~30-40% Neutral, ~10-20% Negative

### Readability
- Answers written at approximately **middle-school reading level**
- Mean Flesch Reading Ease: 60-70 (Standard)
- Mean Grade Level: 7-9th grade

### Collaborative Filtering
- **User-Item Matrix:** Sparse (~99.9% sparsity)
- **Models:** Item-Item CF, User-User CF, Baseline
- **Evaluation:** RMSE and MAE on rating prediction (1-5 scale)
- CF models show improvement over baseline

---

## 📚 Citations

If you use this dataset, please cite:

```
Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems
Mengting Wan, Julian McAuley
International Conference on Data Mining (ICDM), 2016

Addressing complex and subjective product-related queries with customer reviews
Julian McAuley, Alex Yang
World Wide Web (WWW), 2016
```

---

## 🔧 Utility Scripts

### `scripts/download_amazon_qa_data.py`
Downloads all category files from the Amazon Q&A dataset.

### `scripts/data_utils.py`
Utility functions for:
- Parsing gzipped JSON files
- Loading data into pandas DataFrames
- Converting to strict JSON format
- Loading specific categories

---

## 📝 Notes

- All visualizations are automatically saved in their respective folders
- Processed data files are stored in `data/processed/` for reuse
- The notebooks use the provided `parse()` and `getDF()` functions as specified in the dataset documentation
- Sentiment scores are scaled to 1-5 rating scale for collaborative filtering
- This is a data analysis project for MSDS Semester 3 - Recommender System course

---

## 📄 License

This project uses the Amazon Q&A dataset from Julian McAuley (UCSD). Please refer to the original dataset's license and citation requirements.




