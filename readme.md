# 📰 Fake News Identification

## 📌 Project Overview
This project focuses on detecting **fake and real news articles** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The system preprocesses news text, extracts meaningful features using **TF-IDF**, trains classification models, and deploys them through an interactive **Streamlit web application**.

The goal is to demonstrate an **end-to-end machine learning workflow**, from data preprocessing and visualization to model training, evaluation, and deployment.

---

## 🎯 Objectives
- Preprocess and clean raw news text data  
- Perform exploratory data analysis (EDA) and text visualization  
- Train and evaluate machine learning models for fake news detection  
- Build a user-friendly Streamlit application for real-time prediction  

---

## 🗂️ Project Structure
```
fake-news-identification/
│
├── trained_models/
│   ├── tfidf_vectorizer.pkl
│   ├── Logistic Regression.pkl
│   └── Decision Tree Classifier.pkl
│
├── news_detection.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
The dataset contains news articles labeled as:
- **Real (1)**
- **Fake (0)**

### Columns Used
- `text` → News content  
- `class` → Target label  

Columns such as **title, subject, and date** were removed as they did not significantly contribute to classification.

---

## 🔧 Data Preprocessing
- Removed duplicate and null records  
- Removed punctuation and special characters  
- Converted text to lowercase  
- Removed English stopwords  
- Shuffled dataset to avoid model bias  

---

## 📈 Exploratory Data Analysis
- Class distribution visualization using count plots  
- WordClouds for **Real** and **Fake** news  
- Bar chart of top 20 most frequent words  

---

## 🧠 Models Used

| Model | Description |
|------|------------|
| Logistic Regression | Linear classifier using TF-IDF features |
| Decision Tree Classifier | Non-linear tree-based model |

---

## 🔢 Feature Extraction
- **TF-IDF Vectorizer** was used to convert text into numerical form.

---

## ✅ Model Performance
- **Decision Tree Classifier:** ~99% accuracy  
- **Logistic Regression:** ~98% accuracy  

Both models performed exceptionally well, with the **Decision Tree slightly outperforming Logistic Regression**.

---

## 🚀 Deployment (Streamlit App)
The Streamlit app allows users to:
- Enter news text  
- Select a trained model  
- Get real-time predictions with confidence scores  

### App Features
- Model selection (Decision Tree / Logistic Regression)  
- Text preprocessing inside the app  
- Probability-based confidence display  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/UFAQUE123/fake-news-identification.git
cd fake-news-identification
```

### 2️⃣ Create and Activate Environment
```bash
conda create -n fake_news python=3.9
conda activate fake_news
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- NLTK  
- Scikit-learn  
- Matplotlib, Seaborn, WordCloud  
- Streamlit  
- Joblib  

---

## 📌 Conclusion
This project demonstrates how **NLP and Machine Learning** can be effectively applied to classify news articles as **real or fake**.  
The results highlight the strength of classical ML models combined with proper text preprocessing and feature engineering.
