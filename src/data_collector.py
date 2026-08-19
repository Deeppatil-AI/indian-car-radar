"""
Data Collector and Scraper Module for Indian Car Dataset.
Fetches, cleans, verifies, and organizes car images into train/validation splits.
"""

import os
import io
import time
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

from src.indian_cars_metadata import INDIAN_CAR_CLASSES, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def search_and_download_images(
    query: str,
    output_dir: Path,
    max_images: int = 25,
    delay: float = 0.5
) -> int:
    """
    Downloads images for a query using DuckDuckGo public image API endpoints.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Fetch DDG token
    token_url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}"
    try:
        res = requests.get(token_url, headers=headers, timeout=10)
        res.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not connect to search provider for query '{query}': {e}")
        return 0

    import re
    vqd_match = re.search(r"vqd=([\d-]+)", res.text)
    if not vqd_match:
        vqd_match = re.search(r'vqd=\"([\d-]+)\"', res.text)
    
    if not vqd_match:
        logger.warning(f"Could not extract search token for '{query}'.")
        return 0

    vqd = vqd_match.group(1)
    search_api = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={requests.utils.quote(query)}&vqd={vqd}&f=,,,&p=1"
    
    try:
        resp = requests.get(search_api, headers=headers, timeout=10)
        data = resp.json()
        results = data.get("results", [])
    except Exception as e:
        logger.warning(f"Search API error for '{query}': {e}")
        return 0

    downloaded = 0
    for idx, item in enumerate(results):
        if downloaded >= max_images:
            break
        img_url = item.get("image")
        if not img_url:
            continue

        try:
            img_resp = requests.get(img_url, headers=headers, timeout=8)
            if img_resp.status_code == 200:
                # Verify image is valid PIL image
                img = Image.open(io.BytesIO(img_resp.content))
                img = img.convert("RGB")
                
                # Filter out tiny thumbnails
                if img.width >= 150 and img.height >= 150:
                    filename = f"img_{int(time.time()*1000)}_{downloaded}.jpg"
                    img.save(output_dir / filename, "JPEG", quality=90)
                    downloaded += 1
            time.sleep(delay)
        except Exception:
            continue

    logger.info(f"Downloaded {downloaded} images for '{query}' -> {output_dir}")
    return downloaded


def build_indian_cars_dataset(
    target_classes: Optional[List[str]] = None,
    images_per_class: int = 20,
    val_split: float = 0.2
):
    """
    Builds the dataset for specified or all Indian car classes.
    Organizes files into data/train and data/val.
    """
    classes_to_process = INDIAN_CAR_CLASSES
    if target_classes:
        classes_to_process = [c for c in INDIAN_CAR_CLASSES if c["id"] in target_classes]

    raw_base = Path("data/raw")
    train_base = Path("data/train")
    val_base = Path("data/val")

    for car in classes_to_process:
        car_id = car["id"]
        class_raw_dir = raw_base / car_id
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        for query in car.get("query_keywords", [car["make"] + " " + car["model"]]):
            search_and_download_images(query, class_raw_dir, max_images=images_per_class)

        # Distribute into train/val
        images = list(class_raw_dir.glob("*.jpg")) + list(class_raw_dir.glob("*.png"))
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))

        train_dir = train_base / car_id
        val_dir = val_base / car_id
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images[:split_idx]:
            dest = train_dir / img_path.name
            if not dest.exists():
                try:
                    Image.open(img_path).save(dest)
                except Exception:
                    pass

        for img_path in images[split_idx:]:
            dest = val_dir / img_path.name
            if not dest.exists():
                try:
                    Image.open(img_path).save(dest)
                except Exception:
                    pass


def create_cyber_sample_image(car_info: dict, output_path: Path):
    """
    Creates a styled schematic reference image for the car class.
    Used for instant offline testing and bootstrap verification.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 300), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)

    # Cyber grid background
    for x in range(0, 400, 20):
        draw.line([(x, 0), (x, 300)], fill=(25, 25, 25), width=1)
    for y in range(0, 300, 20):
        draw.line([(0, y), (400, y)], fill=(25, 25, 25), width=1)

    # Targeting reticle border
    draw.rectangle([(20, 20), (380, 280)], outline=(220, 220, 220), width=2)
    draw.line([(15, 30), (15, 15), (30, 15)], fill=(255, 255, 255), width=3)
    draw.line([(385, 30), (385, 15), (370, 15)], fill=(255, 255, 255), width=3)
    draw.line([(15, 270), (15, 285), (30, 285)], fill=(255, 255, 255), width=3)
    draw.line([(385, 270), (385, 285), (370, 285)], fill=(255, 255, 255), width=3)

    # Vehicle silhouette schematic
    draw.rectangle([(80, 130), (320, 210)], outline=(200, 200, 200), width=2)  # Body
    draw.polygon([(120, 130), (160, 80), (250, 80), (290, 130)], outline=(240, 240, 240), fill=(30, 30, 30))  # Cabin
    draw.ellipse([(110, 190), (160, 240)], outline=(255, 255, 255), width=3)  # Front wheel
    draw.ellipse([(240, 190), (290, 240)], outline=(255, 255, 255), width=3)  # Rear wheel

    # Text headers
    make = car_info["make"].upper()
    model = car_info["model"].upper()
    gen = car_info["generation"]
    
    draw.text((30, 30), f"[ {make} // {model} ]", fill=(255, 255, 255))
    draw.text((30, 50), f"GEN: {gen}", fill=(180, 180, 180))
    draw.text((30, 250), f"SPECS: {car_info.get('power', 'N/A')}", fill=(160, 160, 160))

    img.save(output_path, "JPEG", quality=95)


def bootstrap_sample_dataset():
    """
    Creates guaranteed reference samples in data/train, data/val, and data/samples.
    """
    for car in INDIAN_CAR_CLASSES:
        car_id = car["id"]
        # Sample for quick gallery
        sample_path = Path("data/samples") / f"{car_id}.jpg"
        create_cyber_sample_image(car, sample_path)
        
        # Training/val samples
        for i in range(3):
            train_path = Path("data/train") / car_id / f"ref_train_{i}.jpg"
            create_cyber_sample_image(car, train_path)
        
        val_path = Path("data/val") / car_id / "ref_val_0.jpg"
        create_cyber_sample_image(car, val_path)

    logger.info("Sample reference dataset bootstrapped successfully.")


if __name__ == "__main__":
    bootstrap_sample_dataset()
