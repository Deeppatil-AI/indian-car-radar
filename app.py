"""
FastAPI Server & Web Application for Indian Car Detection.
Powered by Deep ResNet-50 + DINOv2 Hybrid Embeddings & RLHF Online Metric Learning.
"""

import os
import io
import re
import base64
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch

from src.indian_cars_retriever import IndianCarRetrievalEngine

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("[SERVER] Initializing Indian Car Deep Retrieval Engine on GPU...")
    engine = IndianCarRetrievalEngine(catalog_path="data/unified_catalog.json")
    yield


app = FastAPI(title="CYBER-DETECT // Indian Car Classifier", lifespan=lifespan)

# Mount static and dataset directories
app.mount("/static", StaticFiles(directory="static"), name="static")

dataset_img_dir = Path("Indian Car Recommendation System/All car images")
if dataset_img_dir.exists():
    app.mount("/dataset_images", StaticFiles(directory=str(dataset_img_dir)), name="dataset_images")

cars_ds_train = Path("Cars Dataset/train")
if cars_ds_train.exists():
    app.mount("/cars_dataset_images", StaticFiles(directory=str(cars_ds_train)), name="cars_dataset_images")

user_feedback_dir = Path("data/user_feedback_exemplars")
user_feedback_dir.mkdir(parents=True, exist_ok=True)
app.mount("/feedback_images", StaticFiles(directory=str(user_feedback_dir)), name="feedback_images")

sample_dir = Path("data/samples")
if sample_dir.exists():
    app.mount("/data/samples", StaticFiles(directory="data/samples"), name="samples")


class PredictRequest(BaseModel):
    image_data: str


class FeedbackRequest(BaseModel):
    image_data: str
    predicted_idx: int
    correct_idx: int
    is_correct: bool


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = Path("templates/index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Templates not found</h1>", status_code=404)


@app.get("/api/system_info")
async def get_system_info():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Mode"
    catalog_len = len(engine.catalog) if engine else 304
    rl_stats = engine.rl_stats if engine else {}
    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "device": "cuda" if cuda_avail else "cpu",
        "num_classes": int(catalog_len),
        "rl_stats": {k: int(v) for k, v in rl_stats.items()}
    }


@app.get("/api/classes")
async def get_classes():
    if engine and engine.catalog:
        return [
            {
                "id": str(c["id"]),
                "make": str(c["make"]),
                "model": str(c["model"]),
                "full_name": str(c["full_name"]),
                "image_url": str(c["image_url"])
            }
            for c in engine.catalog
        ]
    return []


@app.get("/api/samples")
async def get_samples():
    if not engine or not engine.catalog:
        return []

    priority_keywords = [
        "Thar", "Nexon", "Creta", "Swift", "Fortuner", "Scorpio", "Seltos",
        "XUV700", "City", "Punch", "Baleno", "Innova", "Harrier", "Virtus",
        "BMW 3", "Audi A4", "Mercedes", "Verna", "Jimny", "Porsche 911"
    ]

    selected = []
    seen = set()

    for kw in priority_keywords:
        for idx, car in enumerate(engine.catalog):
            if kw.lower() in car["full_name"].lower() and car["full_name"] not in seen:
                seen.add(car["full_name"])
                selected.append({
                    "id": str(car["id"]),
                    "catalog_idx": int(idx),
                    "make": str(car["make"]),
                    "model": str(car["model"]),
                    "full_name": str(car["full_name"]),
                    "image_url": str(car["image_url"])
                })
                break
        if len(selected) >= 18:
            break

    return selected


@app.post("/api/predict")
async def predict_car(req: PredictRequest):
    global engine
    if engine is None:
        engine = IndianCarRetrievalEngine(catalog_path="data/unified_catalog.json")

    image_data = req.image_data

    # Load image from local dataset URL or base64
    if image_data.startswith("/dataset_images/"):
        filename = image_data.replace("/dataset_images/", "")
        local_path = Path("Indian Car Recommendation System/All car images") / filename
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/cars_dataset_images/"):
        rel_path = image_data.replace("/cars_dataset_images/", "")
        local_path = Path("Cars Dataset/train") / rel_path
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/feedback_images/"):
        rel_path = image_data.replace("/feedback_images/", "")
        local_path = Path("data/user_feedback_exemplars") / rel_path
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/data/samples/"):
        local_path = Path(image_data.lstrip("/"))
        image = Image.open(local_path).convert("RGB")
    elif "base64," in image_data:
        b64_str = image_data.split("base64,")[1]
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    results = engine.search(image, top_k=4)
    best = results["best_match"]

    response_payload = {
        "car_info": {
            "make": str(best["make"]),
            "model": str(best["model"]),
            "generation": str(best["year_span"]),
            "year_span": str(best["year_span"]),
            "reference_image_url": str(best["image_url"]),
            "catalog_idx": int(best["catalog_idx"])
        },
        "confidence": float(best["confidence"]),
        "top_k": [
            {
                "make": str(c["make"]),
                "model": str(c["model"]),
                "year_span": str(c["year_span"]),
                "confidence": float(c["confidence"]),
                "similarity_pct": float(c["similarity_pct"]),
                "image_url": str(c["image_url"]),
                "catalog_idx": int(c["catalog_idx"])
            }
            for c in results["top_k"]
        ],
        "grad_cam_thermal": str(results["grad_cam_thermal"]),
        "grad_cam_cyber": str(results["grad_cam_cyber"]),
        "rl_stats": {k: int(v) for k, v in engine.rl_stats.items()}
    }

    return response_payload


@app.post("/api/feedback")
async def handle_feedback(req: FeedbackRequest):
    global engine
    if engine is None:
        engine = IndianCarRetrievalEngine(catalog_path="data/unified_catalog.json")

    image_data = req.image_data
    if image_data.startswith("/dataset_images/"):
        filename = image_data.replace("/dataset_images/", "")
        local_path = Path("Indian Car Recommendation System/All car images") / filename
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/cars_dataset_images/"):
        rel_path = image_data.replace("/cars_dataset_images/", "")
        local_path = Path("Cars Dataset/train") / rel_path
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/feedback_images/"):
        rel_path = image_data.replace("/feedback_images/", "")
        local_path = Path("data/user_feedback_exemplars") / rel_path
        image = Image.open(local_path).convert("RGB")
    elif image_data.startswith("/data/samples/"):
        local_path = Path(image_data.lstrip("/"))
        image = Image.open(local_path).convert("RGB")
    elif "base64," in image_data:
        b64_str = image_data.split("base64,")[1]
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    res = engine.apply_reinforcement_feedback(
        query_img=image,
        predicted_idx=int(req.predicted_idx),
        correct_idx=int(req.correct_idx),
        is_correct=bool(req.is_correct)
    )
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
