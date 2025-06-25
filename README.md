# 🧠 Liver Cirrhosis Prediction using Machine Learning

This project is aimed at revolutionizing liver care by using advanced machine learning techniques to predict the presence of liver cirrhosis. It includes a trained ML model and a user-friendly Flask web application for live predictions.

---

## 📊 Dataset
The dataset used is stored in `Liver_Cirrhosis_Dataset.xlsx` and contains various clinical features such as:
- Age
- Gender
- Bilirubin levels
- Alkaline Phosphotase
- And other liver function metrics

---

## 🤖 ML Models Used
- Random Forest
- K-Nearest Neighbors
- XGBoost

The best-performing model (saved as `best_model.pkl`) is used for predictions, along with a normalizer (`normalizer.pkl`) to preprocess inputs.

---

## 🌐 Web App Features
- Built using **Flask**
- Takes user input from a form
- Displays whether the patient is likely **Healthy** or **At Risk**

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Amrutha131204/-LiverCirrhosisPrediction1.git
cd LiverCirrhosisPrediction1
