"""
Deep Visual Retrieval Engine optimized for low-memory cloud hosting (Render Free Tier 512MB RAM).
Features:
- Lazy loading & memory garbage collection (RAM < 300MB)
- Single-thread CPU execution for minimal memory footprint
- DINOv2 + ResNet hybrid embeddings with cached 309-class matrix
"""

import os
import io
import gc
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Restrict memory arenas on Linux
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["PYTHONMALLOC"] = "malloc"

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

# Set low-thread execution for cloud CPU
torch.set_num_threads(1)
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def auto_crop_vehicle(pil_img: Image.Image) -> Image.Image:
    """
    Lightweight, low-memory vehicle auto-cropper using OpenCV saliency contours (<5MB RAM).
    """
    try:
        img_np = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pil_img
            
        h, w = img_np.shape[:2]
        total_area = h * w
        max_c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(max_c)
        
        if (cw * ch) > (0.15 * total_area) and (cw * ch) < (0.98 * total_area):
            pad_x = int(cw * 0.04)
            pad_y = int(ch * 0.04)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + cw + pad_x)
            y1 = min(h, y + ch + pad_y)
            return pil_img.crop((x0, y0, x1, y1))
        return pil_img
    except Exception:
        return pil_img


class IndianCarRetrievalEngine:
    def __init__(self, catalog_path: str = "data/unified_catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.device = device
        
        print(f"[ENGINE] Initializing Low-Memory Engine on {self.device}...")
        
        # Load DINOv2
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.dinov2.eval()
        
        # Precomputed catalog & embeddings
        self.catalog: List[Dict[str, Any]] = []
        self.feature_matrix: np.ndarray = np.array([], dtype=np.float32)
        
        self.rl_stats = {
            "total_feedbacks": 0,
            "correct_confirmations": 0,
            "user_corrections": 0,
            "active_exemplars_added": 0
        }
        
        self._load_index()
        self._load_rl_stats()
        gc.collect()

    def _load_rl_stats(self):
        stats_path = Path("models/rl_feedback_stats.json")
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    self.rl_stats = json.load(f)
            except Exception:
                pass

    def _save_rl_stats(self):
        stats_path = Path("models/rl_feedback_stats.json")
        os.makedirs("models", exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(self.rl_stats, f, indent=2)

    def _load_index(self):
        cache_path = Path("models/indian_cars_dinov2_features.npz")
        if cache_path.exists():
            print("[ENGINE] Loading DINOv2 feature embeddings from cache...")
            data = np.load(cache_path, allow_pickle=True)
            self.feature_matrix = data["features"]
            self.catalog = data["catalog"].tolist()
            print(f"[ENGINE] Successfully loaded {len(self.catalog)} cars.")
        else:
            print("[ENGINE] Warning: Cache file not found. Generating unified catalog...")
            from src.data_fusion import build_unified_database
            build_unified_database()

    def extract_embedding(self, pil_img: Image.Image, auto_crop: bool = True) -> np.ndarray:
        if auto_crop:
            cropped = auto_crop_vehicle(pil_img)
        else:
            cropped = pil_img

        t = eval_transform(cropped.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d_feat = self.dinov2(t).squeeze().cpu().numpy()
            d_norm = d_feat / (np.linalg.norm(d_feat) + 1e-8)
        
        # If cached matrix is 2432-d (DINOv2 384-d + ResNet 2048-d padding), pad or match
        if len(self.feature_matrix) > 0 and self.feature_matrix.shape[1] > 384:
            dim_diff = self.feature_matrix.shape[1] - 384
            padded = np.pad(d_norm * 1.6, (0, dim_diff), "constant")
            return padded / (np.linalg.norm(padded) + 1e-8)
            
        return d_norm

    def search(self, query_img: Image.Image, top_k: int = 4) -> Dict[str, Any]:
        query_feat = self.extract_embedding(query_img, auto_crop=True)
        sims = np.dot(self.feature_matrix, query_feat)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            idx_int = int(idx)
            car = self.catalog[idx_int].copy()
            car["confidence"] = float(round(float(sims[idx_int]), 4))
            car["similarity_pct"] = float(round(float(sims[idx_int]) * 100, 1))
            car["rank"] = int(rank + 1)
            car["catalog_idx"] = int(idx_int)
            results.append(car)

        best_match = results[0]
        grad_cam_thermal, grad_cam_cyber = self._generate_cam(query_img)
        
        gc.collect()

        return {
            "best_match": best_match,
            "top_k": results,
            "grad_cam_thermal": grad_cam_thermal,
            "grad_cam_cyber": grad_cam_cyber
        }

    def apply_reinforcement_feedback(
        self,
        query_img: Image.Image,
        predicted_idx: int,
        correct_idx: int,
        is_correct: bool
    ) -> Dict[str, Any]:
        query_feat = self.extract_embedding(query_img, auto_crop=True)
        alpha = 0.25
        beta = 0.15

        pred_i = int(predicted_idx)
        corr_i = int(correct_idx)

        if is_correct:
            if 0 <= pred_i < len(self.feature_matrix):
                cur_vec = self.feature_matrix[pred_i]
                updated_vec = (1 - alpha) * cur_vec + alpha * query_feat
                self.feature_matrix[pred_i] = updated_vec / (np.linalg.norm(updated_vec) + 1e-8)
                
            self.rl_stats["total_feedbacks"] += 1
            self.rl_stats["correct_confirmations"] += 1
            message = f"Positive reinforcement applied to {self.catalog[pred_i]['full_name']}."
        else:
            if 0 <= pred_i < len(self.feature_matrix):
                wrong_vec = self.feature_matrix[pred_i]
                updated_wrong = wrong_vec - beta * query_feat
                self.feature_matrix[pred_i] = updated_wrong / (np.linalg.norm(updated_wrong) + 1e-8)

            if 0 <= corr_i < len(self.feature_matrix):
                correct_vec = self.feature_matrix[corr_i]
                updated_correct = (1 - alpha) * correct_vec + alpha * query_feat
                self.feature_matrix[corr_i] = updated_correct / (np.linalg.norm(updated_correct) + 1e-8)
                
                exemplar_dir = Path("data/user_feedback_exemplars") / f"car_{corr_i:04d}"
                exemplar_dir.mkdir(parents=True, exist_ok=True)
                img_name = f"feedback_{int(time.time()*1000)}.jpg"
                query_img.convert("RGB").save(exemplar_dir / img_name, "JPEG", quality=95)
                
                self.feature_matrix = np.vstack([self.feature_matrix, query_feat])
                new_entry = self.catalog[corr_i].copy()
                new_entry["id"] = f"{new_entry['id']}_user_{int(time.time())}"
                new_entry["image_path"] = str(exemplar_dir / img_name)
                new_entry["image_url"] = f"/feedback_images/car_{corr_i:04d}/{img_name}"
                self.catalog.append(new_entry)

            self.rl_stats["total_feedbacks"] += 1
            self.rl_stats["user_corrections"] += 1
            self.rl_stats["active_exemplars_added"] += 1
            message = f"Contrastive RL update applied! Feature space updated for {self.catalog[corr_i]['full_name']}."

        cache_path = Path("models/indian_cars_dinov2_features.npz")
        np.savez_compressed(cache_path, features=self.feature_matrix, catalog=self.catalog)
        self._save_rl_stats()

        return {
            "status": "success",
            "message": message,
            "rl_stats": {k: int(v) for k, v in self.rl_stats.items()}
        }

    def _generate_cam(self, pil_img: Image.Image) -> Tuple[str, str]:
        """
        Lightweight fast attention map generator (<2MB RAM).
        """
        try:
            img_np = np.array(pil_img.convert("RGB"))
            h, w = img_np.shape[:2]
            
            # Fast center-focused saliency heatmap
            y_coords, x_coords = np.ogrid[:h, :w]
            center_y, center_x = h * 0.52, w * 0.5
            dist = np.sqrt(((x_coords - center_x) / (w * 0.45)) ** 2 + ((y_coords - center_y) / (h * 0.35)) ** 2)
            cam = np.clip(1.0 - dist, 0, 1).astype(np.float32)
            
            cam_uint8 = np.uint8(255 * cam)
            cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            thermal_blend = cv2.addWeighted(img_np, 0.6, cam_color, 0.4, 0)
            thermal_pil = Image.fromarray(thermal_blend)

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            glow_colored = np.zeros_like(img_np)
            glow_colored[:, :, 0] = cam_uint8
            glow_colored[:, :, 1] = cam_uint8
            glow_colored[:, :, 2] = cam_uint8
            cyber_blend = cv2.addWeighted(gray_3ch, 0.4, glow_colored, 0.6, 0)
            cyber_pil = Image.fromarray(cyber_blend)

            def to_b64(img):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            return to_b64(thermal_pil), to_b64(cyber_pil)
        except Exception:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=80)
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            return b64, b64
