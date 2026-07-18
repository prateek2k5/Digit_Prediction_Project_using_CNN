# ✍️ Handwritten Digit Recognition using CNN (MNIST)

A deep learning project that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The project includes an interactive **Streamlit** web application for real-time digit prediction with confidence scores and probability visualization.

---

## 🚀 Live Demo

🌐 **Try the application here:**

https://digitpredictionprojectusingcnn-m8pqbntmje8jr8tkx2wrgj.streamlit.app/

---

## ✨ Features

- 🧠 CNN-based handwritten digit recognition
- 📊 Real-time digit prediction
- 📈 Prediction confidence score
- 📉 Probability distribution graph
- 🎨 Interactive Streamlit interface
- ⚡ Fast and lightweight inference
- 📱 Simple and user-friendly UI

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib

---

## 📂 Project Structure

```text
Handwritten-Digit-Recognition-MNIST-CNN/
│
├── CNN_Training.ipynb
├── CNN_app.py
├── cnn_model.keras
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 📚 Dataset

The model is trained on the **MNIST Handwritten Digit Dataset**, which contains:

- 📌 60,000 Training Images
- 📌 10,000 Testing Images
- 📌 10 Classes (Digits 0–9)
- 📌 28 × 28 Grayscale Images

---

## 🧠 CNN Architecture

The Convolutional Neural Network consists of:

- Conv2D (32 Filters)
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten Layer
- Dense (256 Neurons)
- Dropout Layer
- Dense (10 Neurons with Softmax)

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/prateek2k5/Handwritten-Digit-Recognition-MNIST-CNN.git
```

### 2️⃣ Navigate to the Project Directory

```bash
cd Handwritten-Digit-Recognition-MNIST-CNN
```

### 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 4️⃣ Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Run the Application

```bash
streamlit run CNN_app.py
```

---

# ⚠️ Important

This project is built and tested using **Python 3.11**.

If you are using **VS Code**, you **must manually select the Python 3.11 interpreter** before running the project.

Go to:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ Select Python 3.11
```

Using Python **3.13 or newer** may cause TensorFlow compatibility issues and prevent the project from running.

---

## 📊 Model Output

The application displays:

- ✅ Predicted Digit
- ✅ Confidence Score
- ✅ Probability Distribution Graph
- ✅ Class-wise Prediction Probabilities

---

## 📸 Application Preview

You can test the deployed application using the Streamlit link above.

---

## 🚀 Future Improvements

- Support custom handwritten image uploads
- Drawing canvas input
- Camera input
- Advanced image preprocessing
- Mobile-responsive interface
- Better visualization dashboard

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## ⭐ Support

If you found this project helpful, please consider giving it a **Star ⭐** on GitHub. Your support is greatly appreciated!

---

## 👨‍💻 Developed By

### **Prateek**

**B.Tech Information Technology | AI & Machine Learning Enthusiast | Software Developer**

Thank you for visiting this repository! 🚀
