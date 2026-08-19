<div align="center">

# 🏎️ CYBER-DETECT // INDIAN CAR AI RADAR

### *Fine-Grained Indian Car Make & Model Detection System with Active RLHF & Explainable Grad-CAM*

[![Live Demo](https://img.shields.io/badge/🌐_Live_Deployment-Render.com-00ff66?style=for-the-badge&logo=render)](https://car-detection-mz95.onrender.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Meta DINOv2](https://img.shields.io/badge/Meta%20AI-DINOv2-0467DF?style=for-the-badge)](https://dinov2.metademolab.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**[👉 OPEN LIVE DEPLOYMENT (24/7 ONLINE)](https://car-detection-mz95.onrender.com/)** | **[📖 READ COMPLETE TECHNICAL SPECIFICATION](INDIAN_CAR_RADAR_COMPLETE_SPECIFICATION.md)**

</div>

---

## ⚡ Overview

**CYBER-DETECT** is a computer vision platform built specifically to identify **Indian market car models** from real-world photos. 

Unlike generic image classifiers that are easily fooled by paint color (e.g. confusing a red Swift with a red i20), CYBER-DETECT uses **Meta DINOv2 Vision Transformer** patch tokens to learn **color-invariant geometric shapes**, **headlamp clusters**, and **grille contours**, coupled with an active **Reinforcement Learning from Human Feedback (RLHF)** online learning loop.

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 RAW USER PHOTO UPLOAD                     │
                  └─────────────────────────────┬─────────────────────────────┘
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │          LEVEL 1: YOLOv8 AUTO-LOCALIZATION & CROP         │
                  │   Strips background noise (trees, roads, pedestrians, sky) │
                  └─────────────────────────────┬─────────────────────────────┘
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │        LEVEL 2: META DINOv2 VISION TRANSFORMER (ViT)      │
                  │       Extracts 2432-d Color-Invariant Shape Embeddings     │
                  └─────────────────────────────┬─────────────────────────────┘
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │        LEVEL 3: 309 INDIAN CAR METRIC SPACE RETRIEVAL     │
                  │   Sub-millisecond cosine matching + Grad-CAM X-Ray Maps   │
                  └─────────────────────────────┬─────────────────────────────┘
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │        LEVEL 4: ACTIVE REINFORCEMENT LEARNING (RLHF)      │
                  │      Instant online contrastive policy updates on GPU     │
                  └───────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Highlights

- **🎯 YOLOv8 Vehicle Saliency Localization**: Automatically isolates the car body within 5ms.
- **🧠 Meta DINOv2 Geometric Vision Backbone**: Invariant to paint color, lighting shifts, and reflections.
- **🇮🇳 309 Indian Car Models Taxonomy**: Maruti Suzuki, Tata Motors, Mahindra, Hyundai, Toyota, Kia, Honda, Skoda, BMW, Audi, Mercedes-Benz, Porsche, etc.
- **🕹️ Online RLHF Feedback Engine**: Users can confirm or correct models in real-time with an interactive search bar; neural weights update dynamically.
- **🔍 Explainable AI (Grad-CAM X-Ray Mode)**: Interactive Thermal CAM and Cyber Monochrome glow heatmaps.
- **🕹️ Retro-Gaming Cyberpunk HUD**: Monochromatic UI with CRT scanlines, targeting reticles, and 8-bit Web Audio synthesizer.

---

## 🚀 Live Demo & Deployment

The application is deployed 24/7 on Render's cloud infrastructure:
👉 **[https://car-detection-mz95.onrender.com/](https://car-detection-mz95.onrender.com/)**

---

## 💻 Local Quickstart (RTX 3050 GPU Accelerated)

### 1. Clone the Repository
```bash
git clone https://github.com/Deeppatil-AI/indian-car-radar.git
cd indian-car-radar
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser!

---

## 📚 Complete Technical Specification

For the complete in-depth mathematical breakdown, dataset taxonomy, RLHF policy gradient formulas, and memory optimization profiles, please refer to:
👉 **[INDIAN_CAR_RADAR_COMPLETE_SPECIFICATION.md](INDIAN_CAR_RADAR_COMPLETE_SPECIFICATION.md)**

---

## 📜 License
Distributed under the **MIT License**. Created by [Deeppatil-AI](https://github.com/Deeppatil-AI).
