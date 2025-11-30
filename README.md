
# Sentiment Analysis Project (TF-IDF + Logistic Regression)

This project performs **sentiment analysis** on a dataset of customer reviews using **Natural Language Processing (NLP)** techniques.  
The model uses **TF-IDF vectorization** and **Logistic Regression** to classify reviews as **positive** or **negative**.

---

## 📌 Project Overview

This project includes:
- Loading a dataset of customer reviews  
- Detecting columns automatically (text + rating/sentiment)  
- Text preprocessing (cleaning, tokenization, lemmatization, stopwords removal)  
- Converting ratings (1–5) to binary sentiment  
- TF-IDF vectorization  
- Logistic Regression model training  
- Hyperparameter tuning with GridSearchCV  
- Evaluation (accuracy, confusion matrix, ROC AUC)  
- Saving the trained model  
- Helper function to predict new reviews  

---

## 📁 Files in This Project

| File | Description |
|------|-------------|
| `sentiment_analysis_uploaded_dataset.ipynb` | Full Jupyter Notebook |
| `sentiment_pipeline_uploaded_dataset.pkl` | Saved ML pipeline |
| `your_dataset.csv` | Your uploaded dataset |
| `README.md` | Documentation |

---

## 🧠 Workflow Summary

### **1️⃣ Load Dataset**
- Automatically identifies the review and rating/sentiment columns.

### **2️⃣ Preprocessing Steps**
- Lowercasing  
- Remove URLs, HTML tags  
- Remove special characters  
- Tokenization  
- Stopword removal  
- Lemmatization  

### **3️⃣ Convert Rating → Sentiment (If Rating Exists)**
- 4–5 → Positive (1)  
- 1–2 → Negative (0)  
- 3 removed  
If sentiment column already exists → used as-is.

### **4️⃣ TF-IDF Vectorization**
- Convert text to numeric vectors  
- Unigrams & bigrams  

### **5️⃣ Logistic Regression Model**
- Trained on TF-IDF vectors  
- GridSearchCV optimizes:
  - C parameter  
  - N-gram range  
  - min_df  
  - max_df  

### **6️⃣ Model Evaluation**
Includes:
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC AUC  
- PR Curve  

### **7️⃣ Save Trained Model**
A pipeline containing both TF-IDF + Logistic Regression is saved.

### **8️⃣ Predict New Text**
Example:
```python
predict_texts(["I love this product!", "Worst experience ever."])
```

---

## 📦 Installation

```bash
pip install numpy pandas scikit-learn matplotlib nltk
```

Download NLTK data:
```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

---

## ▶️ Running the Notebook

1. Open the Jupyter Notebook  
2. Run cells from top to bottom  
3. Upload dataset if needed  
4. Train model and view evaluation results  
5. Use prediction helper to test new reviews  

---

## 🙌 Author
README generated for your sentiment analysis project.
