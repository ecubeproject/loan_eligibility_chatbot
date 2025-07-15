# 🏦 Loan Eligibility Chatbot – AI-Powered Approval Prediction with Explainability

A real-time, explainable AI-based chatbot that predicts loan eligibility using applicant data. Built for a multi-state co-operative bank, this tool improves transparency and automates the screening process using a trained Random Forest model. A demo is available via a public Streamlit app.

## 🚀 Demo

👉 **Try the Demo**: [https://loaneligibilitychatbot-ecube-analytics.streamlit.app/](https://loaneligibilitychatbot-ecube-analytics.streamlit.app/)

---

## 📌 Project Overview

The **Loan Eligibility Chatbot** is a web-based machine learning application designed to:
- Accept user inputs (e.g., income, loan term, employment, credit history)
- Instantly predict if a loan will be **Approved** or **Not Approved**
- Display human-readable explanations for the prediction
- Provide visualizations using **SHAP** to increase trust in model decisions

The goal is to make loan decisions more **transparent**, **efficient**, and **interpretable**.

---

## 🛠️ Tech Stack

| Component         | Technology                          |
|------------------|--------------------------------------|
| **Language**      | Python                              |
| **Libraries**     | `scikit-learn`, `pandas`, `numpy`, `shap`, `joblib`, `streamlit` |
| **Model**         | Random Forest (best performer)      |
| **Explainability**| SHAP summary and waterfall plots     |
| **Deployment**    | AWS EC2 + S3 (prod), Streamlit Cloud (demo) |
| **Pipeline**      | Label encoding, imputation, scaling, joblib export |

---

## ⚙️ ML Pipeline Components

1. **Data Preprocessing**  
   - Handling missing values (imputation)  
   - Label encoding for categorical variables  
   - Feature scaling where applicable

2. **Model Selection & Training**  
   - Evaluated Logistic Regression, XGBoost, SVM  
   - Selected **Random Forest** for high accuracy and stability

3. **Model Explainability**  
   - SHAP used for per-instance and global interpretability  
   - Helps users understand the decision process

4. **Deployment**  
   - Enterprise deployment on AWS (with NDA)  
   - Public prototype hosted via Streamlit Cloud

---

## 📈 Project Outcome

- ✅ Deployed in production for automated loan pre-screening at a co-operative bank
- ⏱️ Reduced manual approval delays with real-time response
- 🔍 Increased transparency via model interpretability (SHAP)
- 📱 Lightweight, responsive UI accessible from desktop or mobile
- 🧪 Public prototype built for demonstration and client onboarding

---

## 🔐 Privacy Note

This chatbot was developed using **real-world data** from a US financial institution. The production version is hosted securely under NDA and is not publicly accessible. The [demo version](https://loaneligibilitychatbot-ecube-analytics.streamlit.app/) is based on a sanitized version of the data.

---

## 📂 Folder Structure
---

## 📦 Setup Instructions

```bash
# Clone repo
git clone https://github.com/yourusername/LoanEligibilityChatbot.git
cd LoanEligibilityChatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app/app.py
---
📜 License

This repository is made available for educational and non-commercial use only. Please contact ecube.analytics@gmail.com for licensing or collaboration inquiries.
🙋‍♂️ Author & Contact

Tejas Desai
LinkedIn: [https://www.linkedin.com/in/tejasddesaiindia/]
email: [https://aimldstejas@gmail.com]

