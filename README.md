# Crop Classification and Smart Advisory using Hyperspectral Imaging

An AI-powered crop classification and smart advisory system using Hyperspectral Imaging (HSI), PCA, and deep learning models such as 3D CNN and 3D U-Net. The project classifies crops, predicts crop health status, and provides intelligent agricultural recommendations through both online AI and offline local RAG support.

---

## Features

- Hyperspectral image preprocessing
- PCA-based dimensionality reduction
- Crop classification using deep learning
- Crop health prediction using spectral analysis
- Smart agricultural advisory system
- Online AI + Offline Local RAG support
- Django-based web application
- Real-time prediction pipeline

---

## 🌱 Crop Health Analysis

Crop health is estimated using a vegetation index formula:

```python
(NIR - Red) / (NIR + Red + 1e-8)
```

Where:
- **NIR** = Near Infrared band
- **Red** = Red spectral band

Higher values generally indicate healthier vegetation.

---

## 🤖 Smart Advisory System

The system supports:
- **Online AI Mode** for AI-generated recommendations
- **Offline Local RAG Mode** for local intelligent responses using classified crop data

---

## Technologies Used

- Python
- Django
- PyTorch
- NumPy
- Scikit-learn
- OpenCV
- SQLite3
- HTML/CSS/JavaScript

---

## 📂 Project Structure

```text
HSIProject/
│
├── classifier/
├── data/
├── ml_models/
├── static/
├── templates/
├── uploaded/
├── db.sqlite3
├── manage.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/varshithakarpurapu/crop-classification-smart-advisory-hsi.git

cd crop-classification-smart-advisory-hsi

pip install -r requirements.txt

python manage.py runserver
```
## Results
### Login Page
<img width="1913" height="934" alt="image" src="https://github.com/user-attachments/assets/75a95fc4-9f9f-4186-88ea-b3532fba298f" />

### Home Page
<img width="1919" height="927" alt="image" src="https://github.com/user-attachments/assets/a457b898-0bfb-4ba0-8ac7-ea3beca15b87" />

- Upload HSI dataset and click on **Predict** to generate:
  - Classified crop map
  - Crop health map
  - Crop-wise summary table
  - Crop-wise farming recommendations
### Predict Page
<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/d43443bf-5359-4edc-a197-dee06c024788" />
<img width="1919" height="926" alt="image" src="https://github.com/user-attachments/assets/b25ebeaa-f7ae-45c3-80e4-b32b24271ca2" />

### RAG Chatbot 
<img width="1902" height="924" alt="image" src="https://github.com/user-attachments/assets/bbca32c9-70e3-406f-a1de-b3e614cd1dc2" />


## 👩‍💻 Author

**Karpurapu Varshitha**

AI/ML enthusiast passionate about Deep Learning, Hyperspectral Imaging, and Full Stack Development.
