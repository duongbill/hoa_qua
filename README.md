# 🍎🥕 Fruits & Vegetables Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống nhận diện hoa quả và rau củ sử dụng Deep Learning (CNN) để phân loại 36 loại hoa quả và rau củ khác nhau với độ chính xác cao.

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Dataset](#-dataset)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Kết quả](#-kết-quả)
- [Demo Web App](#-demo-web-app)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Đóng góp](#-đóng-góp)
- [License](#-license)

## 🎯 Giới thiệu

Dự án này xây dựng một hệ thống AI có khả năng nhận diện và phân loại **36 loại hoa quả và rau củ** từ hình ảnh, sử dụng mạng nơ-ron tích chập (Convolutional Neural Network - CNN).

### Mục tiêu

- Xây dựng mô hình Deep Learning để phân loại hình ảnh hoa quả/rau củ
- Đạt độ chính xác cao (>90%) trên tập test
- Triển khai ứng dụng web thân thiện với người dùng
- Cung cấp công cụ trực quan hóa dữ liệu và kết quả huấn luyện

## ✨ Tính năng

### 🤖 Mô hình AI

- ✅ Phân loại 36 classes (10 loại hoa quả + 26 loại rau củ)
- ✅ CNN architecture với Dropout để chống overfitting
- ✅ Training với 3,600 ảnh, validation 360 ảnh
- ✅ Accuracy ~92-95%

### 🌐 Web Application (Streamlit)

- 🏠 **Home**: Trang chủ giới thiệu
- 📖 **About Project**: Thông tin dataset và mô hình
- 📊 **Data Visualization**:
  - Dataset Overview với Pie chart
  - Model Performance metrics
  - Class Distribution analysis
  - Sample Images preview
  - Training History với Learning Curves
- 🔮 **Prediction**: Upload ảnh và nhận kết quả dự đoán

## 📊 Dataset

### Cấu trúc

```
data/
├── train/          # 100 ảnh/class = 3,600 ảnh
├── validation/     # 10 ảnh/class = 360 ảnh
└── test/           # 10 ảnh/class = 360 ảnh
```

### 36 Classes

**🍎 Hoa quả (10 loại):**

- apple, banana, grapes, kiwi, mango
- orange, pear, pineapple, pomegranate, watermelon

**🥕 Rau củ (26 loại):**

- beetroot, bell pepper, cabbage, capsicum, carrot
- cauliflower, chilli pepper, corn, cucumber, eggplant
- garlic, ginger, jalepeno, lemon, lettuce
- onion, paprika, peas, potato, radish
- soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip

### Đặc điểm Dataset

- ✅ **Balanced dataset** - Số lượng ảnh đều nhau cho mỗi class
- ✅ **Image size**: 64x64 pixels, RGB
- ✅ **Total**: 4,320 ảnh

## 🏗️ Kiến trúc mô hình

```python
Model: Sequential CNN

Input Layer:        64x64x3 (RGB images)
├── Conv2D(32)  →  ReLU  →  Conv2D(32)  →  MaxPool  →  Dropout(0.25)
├── Conv2D(64)  →  ReLU  →  Conv2D(64)  →  MaxPool  →  Dropout(0.25)
├── Flatten
├── Dense(512)  →  ReLU
├── Dense(256)  →  ReLU
├── Dropout(0.5)
└── Dense(36)   →  Softmax (Output Layer)

Optimizer:      Adam
Loss Function:  Categorical Crossentropy
Metrics:        Accuracy
Epochs:         32
Batch Size:     32
```

### Hyperparameters

- **Learning Rate**: Default Adam (0.001)
- **Batch Size**: 32
- **Epochs**: 32
- **Dropout**: 0.25 (Conv layers), 0.5 (Dense layer)

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- pip hoặc conda
- 4GB RAM (8GB khuyến nghị)

### Bước 1: Clone repository

```bash
git clone https://github.com/duongbill/hoa_qua.git
cd hoa_qua
```

### Bước 2: Tạo môi trường ảo

```bash
# Sử dụng venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Hoặc sử dụng conda
conda create -n fruits_veg python=3.8
conda activate fruits_veg
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r Fruit_veg_webapp/requirements.txt
```

### Bước 4: Chuẩn bị dữ liệu

- Download dataset và đặt vào thư mục `data/`
- Hoặc sử dụng dataset có sẵn trong repo

### Bước 5: Download mô hình đã train

- Download file `trained_model.h5` từ Google Drive
- Đặt vào thư mục `Fruit_veg_webapp/`

## 📖 Sử dụng

### 1. Training mô hình (từ đầu)

```bash
# Mở Jupyter Notebook
jupyter notebook trainning_hoa_qua.ipynb

# Chạy tất cả cells để train model
```

### 2. Testing mô hình

```bash
# Mở notebook testing
jupyter notebook test_hoa_qua.ipynb
```

### 3. Chạy Web Application

```bash
cd Fruit_veg_webapp
streamlit run main.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### 4. Dự đoán với Python script

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('trained_model.h5')

# Load và preprocess ảnh
img = Image.open('test_image.jpg')
img = img.resize((64, 64))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Dự đoán
prediction = model.predict(img_array)
class_id = np.argmax(prediction)
print(f"Predicted class: {class_id}")
```

## 📈 Kết quả

### Performance Metrics

| Metric       | Train | Validation | Test  |
| ------------ | ----- | ---------- | ----- |
| **Accuracy** | ~95%  | ~93%       | ~92%  |
| **Loss**     | ~0.15 | ~0.20      | ~0.22 |

### Learning Curves

- Training và Validation accuracy hội tụ tốt
- Không có dấu hiệu overfitting nghiêm trọng
- Model ổn định sau epoch 25-30

### Confusion Matrix

- Các class được phân loại tốt
- Một số confusion giữa các loại rau củ tương tự (capsicum vs bell pepper)

## 🖥️ Demo Web App

### Screenshots

**1. Home Page**
![Home](screenshots/home.png)

**2. Data Visualization**
![Visualization](screenshots/visualization.png)

**3. Prediction**
![Prediction](screenshots/prediction.png)

### Features Web App

- 📊 5 tabs visualization với Plotly interactive charts
- 🎨 UI thân thiện, dễ sử dụng
- 📈 Real-time learning curves visualization
- 🖼️ Preview sample images theo từng class
- 🔮 Upload ảnh và dự đoán ngay lập tức

## 📁 Cấu trúc dự án

```
hoa_qua/
├── data/                           # Dataset (git ignored)
│   ├── train/
│   ├── validation/
│   └── test/
├── Fruit_veg_webapp/               # Web application
│   ├── main.py                     # Streamlit app
│   ├── labels.txt                  # 36 class labels
│   ├── requirements.txt            # Dependencies
│   └── trained_model.h5            # Trained model (git ignored)
├── trainning_hoa_qua.ipynb         # Training notebook
├── test_hoa_qua.ipynb              # Testing notebook
├── main.py                         # Main script
├── training_hist.json              # Training history (git ignored)
├── trained_model.h5                # Trained model backup
├── .gitignore                      # Git ignore file
├── README.md                       # Documentation (this file)
└── baocao.docx                     # Report document
```

## 🛠️ Công nghệ sử dụng

### Deep Learning & ML

- **TensorFlow/Keras** - Framework chính cho CNN
- **NumPy** - Xử lý mảng và dữ liệu
- **Scikit-learn** - Preprocessing và metrics

### Data Visualization

- **Matplotlib** - Biểu đồ cơ bản
- **Seaborn** - Statistical visualization
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation

### Web Development

- **Streamlit** - Web framework
- **Pillow (PIL)** - Image processing

### Development Tools

- **Jupyter Notebook** - Interactive development
- **Google Colab** - Cloud training (optional)
- **Git/GitHub** - Version control

## 🔮 Hướng phát triển

### Phase 1: Cải thiện Model

- [ ] Transfer Learning (VGG16, ResNet, EfficientNet)
- [ ] Data Augmentation nâng cao
- [ ] Hyperparameter tuning
- [ ] Ensemble methods

### Phase 2: Mở rộng tính năng

- [ ] Confidence score & Top-3 predictions
- [ ] Batch prediction (multiple images)
- [ ] Camera/Webcam input
- [ ] Nutritional information
- [ ] Recipe suggestions

### Phase 3: Production

- [ ] REST API với FastAPI
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile app (React Native/Flutter)
- [ ] Database integration (MongoDB/PostgreSQL)

## 🤝 Đóng góp

Contributions, issues và feature requests đều được chào đón!

1. Fork dự án
2. Tạo branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 👨‍💻 Tác giả

**Dương Bill**

- GitHub: [@duongbill](https://github.com/duongbill)
- Email: your.email@example.com

## 📝 License

This project is [MIT](LICENSE) licensed.

## 🙏 Acknowledgments

- Dataset from Kaggle/GitHub Community
- TensorFlow & Keras Documentation
- Streamlit Community
- All contributors and supporters

---

⭐ **Nếu bạn thấy dự án hữu ích, hãy cho một star!** ⭐
