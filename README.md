# Pneumonia-Detection-System

Deep Learning based Pneumonia Detection using Chest X-ray images with Grad-CAM visualization.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

---

# 📌 Overview

This project uses Deep Learning techniques to classify Chest X-ray images as **Pneumonia** or **Normal**.

The system was built using:
- EfficientNetB0
- ResNet50
- Transfer Learning
- Grad-CAM Visualization

The goal of the project is to explore explainable AI techniques in medical image analysis.

---

# 🚀 Features

✅ Pneumonia Detection using Chest X-rays  
✅ Transfer Learning with EfficientNetB0 & ResNet50  
✅ Ensemble Prediction  
✅ Grad-CAM Heatmap Visualization  
✅ Confusion Matrix & ROC Curve  
✅ Explainable AI for Medical Imaging  

---

# 🛠️ Tech Stack

## Languages & Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

# 🧠 Model Architecture

## EfficientNetB0
- Pretrained on ImageNet
- Fine-tuned for binary classification
- Used for feature extraction

## ResNet50
- Residual Learning Architecture
- Used for comparative analysis

## Grad-CAM
Used to visualize the important regions in Chest X-rays that influence predictions.

---

# 📂 Project Structure

```bash
Pneumonia-Detection-System/
│
├── train_model.py
├── gradcam.py
├── evaluate_model.py
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── gradcam_output.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── dataset/
```

---

# 📊 Results

| Model | Accuracy |
|------|------|
| EfficientNetB0 | 95% |
| ResNet50 | 93% |

- Ensemble learning improved overall performance.
- Grad-CAM successfully highlighted infected lung regions.
- The project focuses on explainability in AI-based diagnosis systems.

> Note: Some visual outputs are representative examples used for demonstration purposes.

---

# 🔥 Grad-CAM Visualization

Grad-CAM helps visualize which regions of the Chest X-ray contributed most to the model’s prediction.

## Example Output

Add your image here:

```md
![GradCAM Output](screenshots/gradcam_output.png)
```

---

# 📈 Evaluation Metrics

Add your graphs here:

```md
![Confusion Matrix](screenshots/confusion_matrix.png)

![ROC Curve](screenshots/roc_curve.png)
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Pneumonia-Detection-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Train model:

```bash
python train_model.py
```

Generate Grad-CAM:

```bash
python gradcam.py
```

Evaluate model:

```bash
python evaluate_model.py
```

---

# 🚀 Future Improvements

- Deploy using Streamlit
- Real-time prediction UI
- Multi-disease classification
- Cloud deployment
- Improved dataset balancing

---

# 👨‍💻 Author

## Abhinav Ramesh

AI & Data Science Student passionate about:
- Artificial Intelligence
- Deep Learning
- Full Stack Development
- Explainable AI

---
