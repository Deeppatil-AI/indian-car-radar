# 🚀 Indian Car AI Radar: Deployment & Data Architecture Guide

This guide explains **where all your data is stored**, how the models and embeddings are indexed, and **how to deploy the website online** to free cloud hosting platforms (Hugging Face Spaces, Render, Railway, or Docker).

---

## 🗄️ 1. Where All Data is Saved in the Project

All datasets, images, and model embeddings are organized inside your project directory:

```
C:\Car-Detection\
│
├── All_cars_dataset.csv                   # Full Indian car specifications (Price, Mileage, Ground Clearance, Engine)
├── data/
│   └── unified_catalog.json               # Master catalog connecting 304 cars to images & specs
│
├── Indian Car Recommendation System/
│   └── All car images/                    # 297 high-resolution Indian car photos
│
├── Cars Dataset/                          # 4,165 real images categorized by train/test splits
│   ├── train/                             # Classes: Swift, Creta, Scorpio, Safari, Innova, Audi, Rolls Royce
│   └── test/
│
├── models/
│   ├── indian_cars_features_unified.npz   # Precomputed ResNet-50 2048-dim deep visual embeddings
│   └── best_indian_car_model.pth          # PyTorch checkpoint
│
└── static/ & templates/                   # Frontend Black & White Retro-Gaming Web Application
```

---

## 🌐 2. Deployment Options (Free Cloud Platforms)

### Option A: Deploy to Hugging Face Spaces (Free & Recommended for ML Apps)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **FastAPI / Docker** or **Gradio** as the SDK.
3. Push or upload your `Car-Detection` folder to the repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/indian-car-radar
   git add .
   git commit -m "Deploy Indian Car AI Radar"
   git push hf main
   ```
4. Your website will be live at `https://YOUR_USERNAME-indian-car-radar.hf.space`!

---

### Option B: Deploy with Docker
Your project contains a production-ready `Dockerfile`:
```bash
# 1. Build the Docker container
docker build -t indian-car-radar .

# 2. Run locally or on any cloud server
docker run -p 8000:8000 indian-car-radar
```

---

### Option C: Deploy to Render.com (Free Web Service)
1. Push your repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **"New Web Service"**.
3. Connect your GitHub repository.
4. Render will automatically detect `render.yaml` and `Procfile` and deploy your app for free!

---

## ⚡ 3. Running Locally on Your Machine

```powershell
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser!
