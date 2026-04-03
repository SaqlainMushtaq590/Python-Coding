# 🚔 ArrestIQ — Police Stop Outcome Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)

> An end-to-end Machine Learning web application that predicts the likelihood of arrest during a police traffic stop — built on real-world data with 65,000+ records.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How It Works](#how-it-works)
- [Key Challenges Solved](#key-challenges-solved)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## 📖 About the Project

**ArrestIQ** is a Streamlit-based ML web application that takes traffic stop details as input and predicts whether an arrest is likely to occur. The model is trained on a real-world police stop dataset containing over **65,000 records**.

The goal of this project is to demonstrate how Machine Learning can be applied to public safety data — while also addressing real challenges like **class imbalance**, **model bias**, and **real-world usability**.

---

## ✨ Features

- 🔍 **Arrest Prediction** — Predicts arrest likelihood based on stop details
- 🎯 **Risk Level Display** — Shows HIGH / MODERATE / LOW-MODERATE / LOW risk
- 💡 **Reason Explanation** — Tells the user *why* the prediction was made
- 📋 **Stop Condition Summary** — Visual table showing each factor and its impact
- 📊 **Probability Breakdown** — Full probability scores with expandable details
- ⚖️ **Combined Condition Logic** — Search + Drugs together give more realistic results

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing & analysis |
| Scikit-Learn | ML model (Logistic Regression) |
| Streamlit | Web application frontend |
| Pickle | Model saving & loading |
| Jupyter Notebook | Model training & EDA |

---

## 📁 Project Structure

```
ArrestIQ/
│
├── PoliceApp.py                 # Main Streamlit web application
├── police_arrest_model.pkl      # Trained ML model
├── Police_Data.csv              # Dataset (65,000+ records)
├── Project.ipynb                # Jupyter Notebook (EDA + Model Training)
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ArrestIQ.git
cd ArrestIQ
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

### 3. Run the App
```bash
streamlit run PoliceApp.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 🧠 How It Works

### Input Features:
| Feature | Description |
|---|---|
| Driver Age | Age of the driver (16–90) |
| Gender | Male or Female |
| Race | White / Black / Hispanic / Other |
| Search Conducted | Was the driver searched? |
| Drugs Related Stop | Was the stop drugs-related? |

### Prediction Logic:
```
Search = Yes  AND  Drugs = Yes   →   HIGH Risk      →  Always Arrest Likely
Search = Yes  AND  Drugs = No    →   MODERATE Risk  →  Arrest if model > 15%
Search = No   AND  Drugs = Yes   →   LOW-MODERATE   →  Arrest if model > 25%
Search = No   AND  Drugs = No    →   LOW Risk        →  Arrest if model > 50%
```

---

## 🏆 Key Challenges Solved

### 1. Severe Class Imbalance
The dataset had only **3.7% arrest records** out of 65,000+ stops. A plain model would just always predict "No Arrest" and still show 96% accuracy.

**Solution:** Retrained with `class_weight='balanced'` so the model properly learns the minority class.

### 2. Model Over-Predicting "No Arrest"
Default threshold of 0.5 was too high for an imbalanced dataset.

**Solution:** Implemented dynamic thresholds based on combined conditions (Search + Drugs).

### 3. Feature Dominance
`search_conducted` was dominating all other features with a coefficient of 3.32.

**Solution:** Built combined condition logic so that Search AND Drugs together give the most realistic predictions rather than relying on a single feature.

### 4. Code Bugs Fixed
- Typo: `"Hisponic"` → `"Hispanic"` (feature was always 0)
- Case mismatch: `"other"` → `"Other"`
- Incomplete output message fixed
- Threshold tuned from 30% to realistic values per risk level

---

## ⚠️ Current Limitations

- Model uses only 5 input features — real-world arrest depends on many more factors
- `violation` type (DUI vs speeding) is not yet included
- `stop_duration` is not yet used as a feature
- Dataset may contain historical bias which affects predictions
- Logistic Regression is a simple model — more powerful models not yet tested

---

## 🚀 Roadmap

- [ ] Add `violation` type as a feature (DUI, speeding, equipment etc.)
- [ ] Add `stop_duration` as a feature
- [ ] Implement **SMOTE** for better class imbalance handling
- [ ] Upgrade model to **Random Forest** or **Gradient Boosting**
- [ ] Add ROC curve for proper threshold selection
- [ ] Add model performance dashboard inside the app
- [ ] Deploy on **Streamlit Cloud** for public access
- [ ] Make fully real-world useable InshAllah 🙏

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Saqlain Mushtaq**
- LinkedIn: [My-linkedin](https://www.linkedin.com/in/saqlain-mushtaq-844312380/)
- GitHub: [My-github](https://github.com/SaqlainMushtaq590)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
