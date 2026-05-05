# 🚀 Quick Start Guide

## Directory Structure Overview

```
Project/
├── 📓 notebooks/              # All Jupyter notebooks
├── 📊 visualizations/         # All plots organized by type
├── 💾 data/                   # Raw and processed data
├── 🐍 scripts/                # Utility scripts
├── 📈 results/                # Analysis results (CSV, reports)
└── 📄 README.md              # Detailed documentation
```

---

## 🎯 Quick Run Instructions

### Step 1: Navigate to Project Directory
```bash
cd "/home/tk-lpt-0806/Desktop/MSDS/SEM3/Recommender System/Project"
```

### Step 2: Install Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install vaderSentiment textstat
```

### Step 3: Run the Notebooks

#### Option A: Using Jupyter Lab
```bash
jupyter lab
```
Then open:
1. `notebooks/eda_amazon_qa.ipynb`
2. `notebooks/sentiment_cf_analysis.ipynb`

#### Option B: Using Jupyter Notebook
```bash
cd notebooks
jupyter notebook
```

---

## 📊 Expected Outputs

### After Running EDA Notebook:
- **Data File:** `data/processed/processed_home_kitchen_qa.pkl`
- **Visualizations:** `visualizations/eda/` (8 PNG files)
  - Question type distribution
  - Answer type distribution
  - ASIN distribution
  - Text length analysis (4 plots)
  - Temporal analysis

### After Running Sentiment/CF Notebook:
- **Data File:** `data/processed/enhanced_home_kitchen_qa.pkl`
- **Results File:** `results/cf_evaluation_results.csv`
- **Visualizations:**
  - `visualizations/sentiment/` (2 PNG files)
  - `visualizations/cf/` (5 PNG files)

---

## 🔍 What Each Notebook Does

### 1️⃣ EDA Notebook (`eda_amazon_qa.ipynb`)
- ✅ Loads raw data from `data/raw/qa_Home_and_Kitchen.json.gz`
- ✅ Analyzes question types, products, text lengths
- ✅ Generates temporal trends
- ✅ Saves processed data with text features

### 2️⃣ Sentiment/CF Notebook (`sentiment_cf_analysis.ipynb`)
- ✅ Loads processed data from EDA
- ✅ Performs VADER sentiment analysis
- ✅ Calculates readability scores
- ✅ Extracts lexical features
- ✅ Builds user-item matrix
- ✅ Implements Item-Item and User-User CF
- ✅ Evaluates with RMSE/MAE

---

## 📁 File Locations

### Data Files
```
data/
├── raw/
│   └── qa_Home_and_Kitchen.json.gz        # Original dataset (17 MB)
└── processed/
    ├── processed_home_kitchen_qa.pkl      # After EDA
    └── enhanced_home_kitchen_qa.pkl       # After Sentiment/CF
```

### Notebooks
```
notebooks/
├── eda_amazon_qa.ipynb                    # Run this FIRST
└── sentiment_cf_analysis.ipynb            # Run this SECOND
```

### Visualizations
```
visualizations/
├── eda/                                   # 8 EDA plots
├── sentiment/                             # 2 sentiment plots
└── cf/                                    # 5 CF plots
```

### Results
```
results/
└── cf_evaluation_results.csv              # Model comparison metrics
```

---

## ⚡ Running Order

**IMPORTANT:** Run notebooks in this order:

1. **First:** `notebooks/eda_amazon_qa.ipynb`
   - Creates: `data/processed/processed_home_kitchen_qa.pkl`
   - Time: ~5-10 minutes

2. **Second:** `notebooks/sentiment_cf_analysis.ipynb`
   - Requires: Output from EDA notebook
   - Creates: `data/processed/enhanced_home_kitchen_qa.pkl`
   - Time: ~10-15 minutes (sentiment analysis is slow)

---

## 🐛 Troubleshooting

### Issue: "File not found" errors
**Solution:** Make sure you're running notebooks from the `notebooks/` directory
```bash
cd notebooks
jupyter notebook eda_amazon_qa.ipynb
```

### Issue: Missing packages
**Solution:** Install missing packages
```bash
pip install vaderSentiment textstat
```

### Issue: "processed_home_kitchen_qa.pkl not found"
**Solution:** Run the EDA notebook first! The sentiment/CF notebook depends on it.

---

## 📊 Dataset Statistics

- **Total Questions:** 184,439
- **Total Products:** ~67,000 unique ASINs
- **Question Types:** Yes/No (60%) and Open-ended (40%)
- **Date Range:** 2002-2018
- **File Size:** 17 MB (compressed)

---

## 💡 Tips

1. **Run cells sequentially** - Don't skip cells
2. **Wait for completion** - Some cells take several minutes
3. **Check outputs** - Verify visualizations are generated
4. **Save work** - Notebooks auto-save, but manually save before closing

---

## 🎓 Key Features Extracted

### From EDA:
- Question/answer word counts
- Question/answer character lengths
- Temporal features (year, month, day of week)

### From Sentiment/CF:
- Sentiment scores (neg, neu, pos, compound)
- Readability scores (Flesch, Gunning Fog, SMOG, etc.)
- Lexical features (punctuation counts)
- User-Item ratings (1-5 scale from sentiment)

---

## 📧 Need Help?

Refer to the detailed `README.md` for more information!

