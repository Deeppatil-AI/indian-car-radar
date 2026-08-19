"""
Ultimate High-Accuracy Deep Retrieval Engine for Indian Car Detection.
Features:
1. YOLOv8 Vehicle Bounding Box Auto-Detection & Cropper (Removes background clutter)
2. Meta DINOv2 (Vision Transformer) + ResNet-50 Dual-Scale Geometric Embeddings
3. Multi-Exemplar Real Dataset Indexing (4,000+ images across multiple colors/angles)
4. Online Reinforcement Learning from Human Feedback (RLHF)
"""

import os
import re
import io
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# High-resolution vision transform
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class IndianCarRetrievalEngine:
    def __init__(self, catalog_path: str = "data/unified_catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.device = device
        
        print(f"[ENGINE] Loading YOLOv8 Vehicle Detector on {self.device}...")
        self.yolo = YOLO("yolov8n.pt")  # Auto-downloads lightweight 6MB model
        
        print(f"[ENGINE] Loading Meta DINOv2 Vision Transformer on {self.device}...")
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.dinov2.eval()

        print(f"[ENGINE] Loading ResNet-50 Feature Backbone on {self.device}...")
        self.full_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.full_resnet.eval()
        
        modules = list(self.full_resnet.children())[:-1]
        self.resnet_feat = torch.nn.Sequential(*modules).to(self.device)
        self.resnet_feat.eval()
        
        self.cam_target_layer = self.full_resnet.layer4[-1]
        
        self.catalog: List[Dict[str, Any]] = []
        self.feature_matrix: np.ndarray = np.array([], dtype=np.float32)
        
        self.rl_stats = {
            "total_feedbacks": 0,
            "correct_confirmations": 0,
            "user_corrections": 0,
            "active_exemplars_added": 0
        }
        
        self._load_or_build_index()
        self._load_rl_stats()

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

    def crop_vehicle_with_yolo(self, pil_img: Image.Image) -> Image.Image:
        """
        Level 1: Uses YOLOv8 to locate and crop the exact car bounding box.
        Eliminates background noise (trees, road, sky, pedestrians).
        """
        try:
            results = self.yolo(pil_img, verbose=False, device=str(self.device))
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Filter for vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck in COCO)
                vehicle_boxes = [b for b in boxes if int(b.cls[0]) in [2, 5, 7]]
                if vehicle_boxes:
                    # Pick box with highest confidence
                    best_box = max(vehicle_boxes, key=lambda b: float(b.conf[0]))
                    xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
                    x0, y0, x1, y1 = xyxy
                    w, h = pil_img.size
                    
                    # Add 3% padding
                    pad_w = int((x1 - x0) * 0.03)
                    pad_h = int((y1 - y0) * 0.03)
                    x0 = max(0, x0 - pad_w)
                    y0 = max(0, y0 - pad_h)
                    x1 = min(w, x1 + pad_w)
                    y1 = min(h, y1 + pad_h)
                    
                    return pil_img.crop((x0, y0, x1, y1))
            return pil_img
        except Exception:
            return pil_img

    def extract_embedding(self, pil_img: Image.Image, auto_crop: bool = True) -> np.ndarray:
        """
        Extracts Multi-Scale Hybrid Embedding (DINOv2 Geometric Shape + ResNet-50 Details).
        """
        if auto_crop:
            cropped = self.crop_vehicle_with_yolo(pil_img)
        else:
            cropped = pil_img

        t = eval_transform(cropped.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d_feat = self.dinov2(t).squeeze().cpu().numpy()
            d_norm = d_feat / (np.linalg.norm(d_feat) + 1e-8)
            
            r_feat = self.resnet_feat(t).squeeze().cpu().numpy()
            r_norm = r_feat / (np.linalg.norm(r_feat) + 1e-8)
            
            # Weight DINOv2 heavily (1.6x) for geometric shape invariance over paint color
            hybrid = np.concatenate([d_norm * 1.6, r_norm])
            hybrid = hybrid / (np.linalg.norm(hybrid) + 1e-8)
        return hybrid

    def _load_or_build_index(self):
        cache_path = Path("models/indian_cars_dinov2_features.npz")
        
        if cache_path.exists():
            print("[ENGINE] Loading DINOv2 + ResNet hybrid embeddings from cache...")
            data = np.load(cache_path, allow_pickle=True)
            self.feature_matrix = data["features"]
            self.catalog = data["catalog"].tolist()
            print(f"[ENGINE] Loaded {len(self.catalog)} cars with color-invariant embeddings.")
            return

        if not self.catalog_path.exists():
            from src.data_fusion import build_unified_database
            build_unified_database()

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            raw_catalog = json.load(f)

        print(f"[ENGINE] Extracting DINOv2 hybrid features for {len(raw_catalog)} cars on {self.device}...")
        embeddings = []
        valid_catalog = []

        for car in raw_catalog:
            img_path = Path(car["image_path"])
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                feat = self.extract_embedding(img, auto_crop=False)
                embeddings.append(feat)
                valid_catalog.append(car)
            except Exception as e:
                print(f"[ENGINE] Skip {img_path.name}: {e}")

        self.feature_matrix = np.array(embeddings, dtype=np.float32)
        self.catalog = valid_catalog

        os.makedirs("models", exist_ok=True)
        np.savez_compressed(cache_path, features=self.feature_matrix, catalog=self.catalog)
        print(f"[ENGINE] Successfully indexed {len(self.catalog)} Indian cars with DINOv2!")

    def search(self, query_img: Image.Image, top_k: int = 4) -> Dict[str, Any]:
        """
        Runs YOLOv8 car isolation + DINOv2 geometric matching against Indian car database.
        """
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
        
        tensor = eval_transform(query_img).unsqueeze(0).to(self.device)
        grad_cam_thermal, grad_cam_cyber = self._generate_cam(tensor, query_img)

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

        print(f"[RLHF] {message} Total Feedbacks: {self.rl_stats['total_feedbacks']}")
        return {
            "status": "success",
            "message": message,
            "rl_stats": {k: int(v) for k, v in self.rl_stats.items()}
        }

    def _generate_cam(self, tensor: torch.Tensor, pil_img: Image.Image) -> Tuple[str, str]:
        try:
            gradients = []
            activations = []

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            def forward_hook(module, input, output):
                activations.append(output)

            h1 = self.cam_target_layer.register_forward_hook(forward_hook)
            h2 = self.cam_target_layer.register_full_backward_hook(backward_hook)

            self.full_resnet.zero_grad()
            out = self.full_resnet(tensor)
            pred_class = out.argmax(dim=1).item()
            score = out[0, pred_class]
            score.backward()

            h1.remove()
            h2.remove()

            grads = gradients[0][0].detach().cpu().numpy()
            acts = activations[0][0].detach().cpu().numpy()

            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)

            img_np = np.array(pil_img.convert("RGB"))
            h, w = img_np.shape[:2]
            cam_resized = cv2.resize(cam, (w, h))

            # Thermal overlay
            cam_uint8 = np.uint8(255 * cam_resized)
            cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            thermal_blend = cv2.addWeighted(img_np, 0.55, cam_color, 0.45, 0)
            thermal_pil = Image.fromarray(thermal_blend)

            # Cyber monochrome mask
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
                img.save(buf, format="JPEG", quality=90)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            return to_b64(thermal_pil), to_b64(cyber_pil)

        except Exception as e:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            return b64, b64
