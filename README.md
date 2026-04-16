# 🌱 DigiFarmer - AI-Powered Agricultural Intelligence

**Transform your farming decisions with cutting-edge AI technology**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)

</div>

## 🎯 Overview

DigiFarmer is a modern, AI-powered web application that revolutionizes agricultural decision-making by combining advanced computer vision and machine learning technologies. Upload a soil image and receive intelligent crop recommendations based on soil classification and environmental analysis.

### ✨ Key Features

- **🔬 Advanced Soil Classification**: ResNet50-powered computer vision for accurate soil type identification
- **🧠 AI Crop Recommendations**: Machine learning algorithms trained on comprehensive agricultural datasets  
- **🌡️ Environmental Analysis**: Multi-factor analysis including N, P, K, temperature, humidity, pH, and rainfall
- **💻 Modern Web Interface**: Responsive, intuitive design with real-time feedback
- **🚀 High-Performance Backend**: FastAPI with automatic API documentation
- **📱 Mobile-First Design**: Seamless experience across all devices

## 🏗️ Project Architecture

```
DigiFarmer/
├── 📁 frontend/                    # Modern web interface
│   ├── 📁 css/
│   │   └── style.css              # Responsive styling with animations
│   ├── 📁 js/
│   │   └── script.js              # Interactive functionality
│   └── 📁 templates/
│       └── index.html             # Single-page application
├── 📁 backend/                     # FastAPI backend services
│   ├── 📁 api/
│   │   └── main.py                # REST API endpoints
│   ├── 📁 models/
│   │   └── combined_crop_soil_recommender.py  # Core AI logic
│   └── 📁 utils/                  # Utility functions
├── 📁 model_outputs/              # Trained ML models
│   ├── soil_classifier_model.keras
│   ├── crop_model.pkl
│   └── crop_label_encoder.pkl
├── 📁 dataset/                    # Training datasets
└── 📄 README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (for model loading)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vlspranay/Crop-Recommendation-System.git
   cd Crop-Recommendation-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Launch the application**
   ```bash
   python run_app.py
   ```

4. **Access the web interface**
   ```
   🌐 Open: http://localhost:8000
   📚 API Docs: http://localhost:8000/api/docs
   ```

## 🎨 User Interface

### Modern Design Features
- **Glassmorphism Effects**: Beautiful translucent cards with backdrop blur
- **Smooth Animations**: Engaging micro-interactions and transitions  
- **Drag & Drop Upload**: Intuitive file handling with visual feedback
- **Real-time Validation**: Instant feedback on user inputs
- **Responsive Layout**: Optimized for desktop, tablet, and mobile

### User Experience
- **One-Click Analysis**: Simple workflow from upload to results
- **Visual Results**: Color-coded confidence scores and recommendations
- **Export Functionality**: Download detailed analysis reports
- **Keyboard Shortcuts**: Power user features (Ctrl+U to upload, Ctrl+Enter to analyze)

## 🔧 API Endpoints

| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/` | GET | Main web interface | < 100ms |
| `/api/complete-analysis` | POST | Full soil analysis + crop recommendations | 2-5s |
| `/api/classify` | POST | Soil classification only | 1-3s |
| `/api/recommend` | POST | Crop recommendations for known soil type | < 500ms |
| `/api/soil-types` | GET | Available soil types and mappings | < 100ms |
| `/api/health` | GET | System health check | < 50ms |
| `/api/stats` | GET | System statistics | < 100ms |


## 🌍 Supported Soil Types & Crops

<details>
<summary><strong>🔍 Click to view detailed soil-crop mappings</strong></summary>

| Soil Type | Characteristics | Best Crops | pH Range |
|-----------|-----------------|------------|----------|
| **Alluvial Soil** | Fertile, well-drained, high organic content | Rice, Wheat, Sugarcane, Cotton, Maize | 6.0-8.0 |
| **Black Soil** | High clay content, moisture-retentive | Cotton, Sugarcane, Wheat, Sunflower | 7.0-8.5 |
| **Cinder Soil** | Volcanic origin, well-drained, porous | Coffee, Tea, Cardamom, Pepper | 5.5-7.0 |
| **Clay Soil** | High water retention, slow drainage | Rice, Wheat, Barley, Potatoes | 6.5-8.0 |
| **Laterite Soil** | Iron-rich, acidic, well-drained | Cashew, Coconut, Rubber, Tea | 5.0-6.5 |
| **Peat Soil** | Organic-rich, acidic, high water content | Rice, Vegetables, Fruits, Herbs | 4.0-6.0 |
| **Red Soil** | Iron oxide content, well-drained | Groundnut, Potato, Rice, Pulses | 5.5-7.5 |
| **Yellow Soil** | Sandy texture, low fertility | Wheat, Barley, Potato, Maize | 6.0-7.5 |

</details>

## 📊 Environmental Parameters

The AI system analyzes these critical factors:

- **🧪 Nutrients**: Nitrogen (20-120 ppm), Phosphorus (10-70 ppm), Potassium (15-90 ppm)
- **🌡️ Climate**: Temperature (10-40°C), Humidity (50-95%)
- **💧 Water**: pH Level (4.0-8.5), Rainfall (50-500 mm)

## 📂 Data & Datasets

The system relies on comprehensive agricultural data stored in JSON format:

### Data Files
- **`crop_requirements_ap.json`**: Detailed crop-specific requirements including:
  - Optimal nutrient levels (N, P, K ranges)
  - Temperature and humidity thresholds
  - pH requirements
  - Rainfall patterns
  - Growing season duration
  - Yield expectations

- **`crop_economics.json`**: Economic information for decision-making:
  - Crop market prices
  - Production costs
  - Profitability metrics
  - Regional price variations
  - Market demand trends

- **`groundwater_ap.json`**: Regional groundwater level data:
  - Available water resources by location
  - Seasonal variations
  - Sustainability indicators
  - Water usage recommendations

### Model Training Data
- **Soil Classification Dataset**: 8 soil types with thousands of annotated soil images
- **Crop Recommendation Dataset**: Historical agricultural records with crop success rates
- **Environmental Data**: Historical weather patterns and crop yields by region

## 🧠 ML Decision System

### How Recommendations Are Generated

#### Step 1: Soil Classification
```
User Image → ResNet50 Deep Learning Model → Soil Type + Confidence Score
```
- **Model**: Fine-tuned ResNet50 with transfer learning
- **Input**: Soil image (224×224 pixels)
- **Output**: Predicted soil type with confidence percentage (0-100%)
- **Classes**: Alluvial, Black, Cinder, Clay, Laterite, Peat, Red, Yellow
- **Accuracy**: 85%+ on test datasets

#### Step 2: Environmental Data Integration
```
Soil Type + Environmental Parameters → Soil-Specific Environmental Ranges
```
- System retrieves optimal ranges for detected soil type
- Validates environmental parameters against healthy ranges
- Flags suboptimal conditions for user awareness

#### Step 3: Crop Recommendation Ranking
```
Soil Type + Environmental Data → Scikit-learn Classifier → Crop Scores
```
- **Model**: Random Forest classifier with 100 estimators
- **Features**: 8 environmental parameters + soil type encoding
- **Ranking**: Crops scored 0-100 based on suitability
- **Decision Logic**:
  1. Filter crops suitable for detected soil type (from soil-crop mapping)
  2. Score each crop based on environmental parameter alignment
  3. Apply economic weighting (market demand, profitability)
  4. Rank by combined ecological and economic scores

#### Step 4: Output Generation
```
Ranked Crops → Confidence Scores → User-Friendly Recommendations
```
- **Primary Recommendations**: Top 3-5 crops with highest scores
- **Confidence Indicators**: Visual feedback based on model certainty
- **Alternative Crops**: Secondary suggestions for diverse farming
- **Warnings**: Alerts for suboptimal environmental conditions

### Decision Factors

| Factor | Weight | Impact |
|--------|--------|--------|
| Soil Type Match | 40% | Primary determinant of crop suitability |
| Nutrient Levels | 20% | Essential for crop growth |
| Temperature Range | 15% | Critical for germination and yield |
| Humidity & Rainfall | 15% | Determines irrigation requirements |
| pH Level | 10% | Affects nutrient availability |

### Confidence & Accuracy Metrics
- **Soil Classification Confidence**: Based on ResNet50 prediction probability
- **Crop Recommendation Score**: 0-100 scale derived from Random Forest prediction confidence
- **Green (80-100)**: Highly recommended - optimal conditions
- **Yellow (60-79)**: Recommended - acceptable conditions
- **Orange (40-59)**: Marginal - requires monitoring
- **Red (<40)**: Not recommended - suboptimal conditions

## 🎯 How to Use

### 1. **Upload Soil Image**
- Drag & drop or click to browse
- Supported: JPG, PNG, WebP (max 10MB)
- Best results: Clear, well-lit soil images

### 2. **Set Environmental Parameters** *(Optional)*
- Fill in known conditions or leave blank for defaults
- Real-time validation ensures valid ranges

### 3. **Analyze & Get Results**
- Click "Analyze Soil & Get Recommendations"
- AI processing takes 2-5 seconds
- Results include confidence scores and suitability ratings

### 4. **Review Recommendations**
- **Soil Analysis**: Type identification with confidence level
- **Environmental Conditions**: Current or default parameters
- **Crop Recommendations**: AI-ranked suggestions with scores
- **Soil-Specific Crops**: Traditional crops for the soil type

## 🔬 Technical Implementation

### Frontend Stack
- **HTML5/CSS3**: Semantic markup with modern styling
- **Vanilla JavaScript**: No framework dependencies, optimized performance
- **CSS Grid/Flexbox**: Responsive layouts
- **CSS Animations**: Smooth transitions and micro-interactions

### Backend Stack
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server with auto-reload
- **CORS**: Cross-origin resource sharing support

### AI/ML Stack
- **TensorFlow**: Deep learning framework for soil classification
- **ResNet50**: Pre-trained CNN with transfer learning
- **Scikit-learn**: Random Forest for crop recommendations
- **NumPy/Pandas**: Data processing and analysis

### Performance Optimizations
- **Model Caching**: Pre-loaded models for fast inference
- **Image Preprocessing**: Optimized pipeline for soil images
- **Async Processing**: Non-blocking request handling
- **Error Handling**: Comprehensive error management

## 📈 Performance Metrics

- **Image Processing**: 2-3 seconds average
- **API Response Time**: < 100ms for most endpoints
- **Concurrent Users**: Supports 100+ simultaneous users
- **Memory Usage**: ~500MB with all models loaded
- **Accuracy**: 85%+ soil classification accuracy

## 🙏 Acknowledgments

- **AICTE** for project framework and support
- **Agricultural Research Community** for soil and crop datasets
- **Open Source Libraries** for ML and web frameworks
- **Contributors** and beta testers for valuable feedback

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support & Contact

- **📧 Email**: [jayanthsrinivas.b@gmail.com](mailto:jayanthsrinivas.b@gmail.com)
- **💬 Discussions**: [GitHub Discussions](https://github.com/vlspranay/Crop-Recommendation-System/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/vlspranay/Crop-Recommendation-System/issues)
- **📚 Repository**: [github.com/vlspranay/Crop-Recommendation-System](https://github.com/vlspranay/Crop-Recommendation-System)

---

<div align="center">

**🌱 Built with ❤️ for the future of agriculture**

[![GitHub Stars](https://img.shields.io/github/stars/vlspranay/Crop-Recommendation-System?style=for-the-badge&logo=github)](https://github.com/vlspranay/Crop-Recommendation-System)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://python.org)
[![Powered by AI](https://img.shields.io/badge/Powered%20by-AI-green?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/github/license/vlspranay/Crop-Recommendation-System?style=for-the-badge)](https://github.com/vlspranay/Crop-Recommendation-System/blob/main/LICENSE)

</div>
