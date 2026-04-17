#  AI Intrusion Detection System

This project is a simple machine learning-based web app that detects whether network activity is normal or an attack.

We built this to understand how intrusion detection systems work and how machine learning can be applied to real-world cybersecurity problems.

---

## What this project does

* Takes network input (like duration, bytes, etc.)
* Uses a trained ML model to classify it
* Shows whether the traffic is **safe or malicious**
* Includes a basic Streamlit UI for interaction

---

## Tech used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib / Seaborn

---

## Project structure

```
app.py → Streamlit UI  
main.py → training + evaluation  
test_model.py → testing  

src/ → all core logic  
```

---

## How to run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the app:

```
streamlit run app.py
```

---

## Model

We used:

* Random Forest
* Decision Tree

Random Forest performed better overall, so it is used in the app.

---

## Dataset

We used the KDD Cup dataset.

It is not included in the repo because of size.

You can download it from:
https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data

After downloading, place it inside:

```
data/KDDTrain+.txt
```

---

## What we learned

* How to structure an ML project properly
* Data preprocessing and feature handling
* Training and evaluating models
* Building a simple UI using Streamlit

---

## Authors

Vimla Pandey
🔗 https://www.linkedin.com/in/vimlapandey/

Vasundhara Raj Mani
🔗 https://www.linkedin.com/in/vashundhara08/
