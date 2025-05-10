# 🧠 Retinal Disease Detection App

AI-powered deep learning app for detecting **Diabetic Retinopathy**, **Glaucoma**, and **Healthy retinas** using retinal fundus images.  
Built as a full-stack **end-to-end machine learning project**, it combines medical insight, deep learning, and cloud-ready deployment with professional-grade reporting.

<p align="center">
  <img src="images/sample_ui.png" alt="App UI Screenshot" width="600">
</p>

---

## 🚀 Live Demo

> 🧪 The app will be hosted soon on Streamlit Cloud for public use.  
> 👉 [Launch Live App (Coming Soon)](#)

---

## ✨ Features

✅ Upload multiple retinal fundus images  
✅ Predict `Diabetic Retinopathy`, `Glaucoma`, or `Healthy`  
✅ Visual confidence bar chart for each prediction  
✅ Dynamic UI feedback (green for healthy, red for risk)  
✅ Auto-generated downloadable **PDF medical report**  
✅ Built with `Streamlit`, `TensorFlow`, and `fpdf2`

---

## 🎯 Motivation

Retinal diseases like diabetic retinopathy and glaucoma are major causes of irreversible blindness — and early detection can change outcomes dramatically. This project simulates an AI-assisted pre-diagnosis tool for clinics or rural health centers with limited specialist access.

---

## 🧠 Model Overview

- **Framework**: TensorFlow + Keras
- **Model Type**: CNN (Convolutional Neural Network)
- **Input Shape**: 224x224 RGB
- **Classes**: Diabetic Retinopathy, Glaucoma, Healthy
- **Output**: Softmax prediction with confidence scores
- **Trained On**: Preprocessed fundus image dataset from [Kaggle](https://www.kaggle.com/)

---

## 🧪 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visuals-yellow?logo=matplotlib)
![FPDF2](https://img.shields.io/badge/FPDF2-PDF%20Reports-green)

---

## 🖥 Local Setup

### Clone the repo & install requirements

```bash
git clone https://github.com/yourusername/retinal-disease-detector.git
cd retinal-disease-detector
pip install -r requirements.txt
