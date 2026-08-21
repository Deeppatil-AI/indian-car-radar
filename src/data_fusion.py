"""
Data Fusion & Deduplication Engine for Indian Cars.
Correctly groups multi-image datasets (Cars Dataset) by parent directory class name.
"""

import os
import re
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any

def standardize_car_name(raw_name: str) -> Tuple[str, str, str]:
    clean = re.sub(r'^\d+_', '', raw_name)
    clean = re.sub(r'^\d{4}-', '', clean).replace('_', ' ').replace('-', ' ').strip()
    clean_lower = clean.lower()
    
    # Maruti Suzuki models
    maruti_models = [
        "swift", "baleno", "alto 800", "alto k10", "alto", "dzire", "brezza",
        "ertiga", "wagon r", "wagonr", "grand vitara", "jimny", "ignis",
        "celerio", "eeco", "ciaz", "fronx", "invicto", "xl6", "s presso", "spresso"
    ]
    for mm in maruti_models:
        if mm in clean_lower:
            model_cap = mm.title().replace("K10", "K10").replace("Xl6", "XL6")
            if model_cap == "Wagonr": model_cap = "Wagon R"
            if model_cap == "Alto": model_cap = "Alto 800"
            return "Maruti Suzuki", model_cap, f"Maruti Suzuki {model_cap}"

    # Tata Motors models
    tata_models = [
        "nexon ev", "nexon", "safari", "harrier", "punch", "altroz",
        "tiago", "tigor", "curvv", "sierra", "indica", "indigo", "hexa", "aria"
    ]
    for tm in tata_models:
        if tm in clean_lower:
            model_cap = tm.title()
            if "Ev" in model_cap: model_cap = model_cap.replace("Ev", "EV")
            return "Tata Motors", model_cap, f"Tata Motors {model_cap}"

    # Mahindra models
    mahindra_models = [
        "thar", "scorpio n", "scorpio classic", "scorpio", "xuv700", "xuv300",
        "xuv3xo", "xuv400", "bolero neo", "bolero", "kuv100", "tuv300", "marazzo", "alturas"
    ]
    for mhm in mahindra_models:
        if mhm in clean_lower:
            model_cap = mhm.upper() if "xuv" in mhm or "kuv" in mhm or "tuv" in mhm else mhm.title()
            if "Scorpio N" in model_cap: model_cap = "Scorpio-N"
            return "Mahindra", model_cap, f"Mahindra {model_cap}"

    # Hyundai models
    hyundai_models = [
        "creta", "venue", "verna", "i20", "i10", "grand i10 nios", "grand i10",
        "tucson", "alcazar", "aura", "exter", "ioniq 5", "kona", "santro"
    ]
    for hm in hyundai_models:
        if hm in clean_lower:
            model_cap = hm.title()
            if "i20" in hm: model_cap = "i20"
            if "i10" in hm: model_cap = "i10"
            return "Hyundai", model_cap, f"Hyundai {model_cap}"

    # Toyota models
    toyota_models = [
        "fortuner legender", "fortuner", "innova hycross", "innova crysta", "innova",
        "glanza", "urban cruiser hyryder", "hyryder", "camry", "hilux", "vellfire", "etios"
    ]
    for tym in toyota_models:
        if tym in clean_lower:
            model_cap = tym.title()
            return "Toyota", model_cap, f"Toyota {model_cap}"

    # Kia models
    kia_models = ["seltos", "sonet", "carens", "ev6", "carnival", "ev9"]
    for km in kia_models:
        if km in clean_lower:
            model_cap = km.title().replace("Ev6", "EV6").replace("Ev9", "EV9")
            return "Kia", model_cap, f"Kia {model_cap}"

    # Honda models
    honda_models = ["city", "amaze", "elevate", "civic", "cr v", "jazz", "wr v", "brio"]
    for hdm in honda_models:
        if hdm in clean_lower:
            model_cap = hdm.title().replace("Wr V", "WR-V").replace("Cr V", "CR-V")
            return "Honda", model_cap, f"Honda {model_cap}"

    # Luxury Brands
    parts = clean.split()
    if len(parts) >= 2:
        p0, p1 = parts[0].lower(), parts[1].lower()
        if p0 == "aston" and p1 == "martin":
            return "Aston Martin", " ".join(parts[2:]).title() or "Vantage", f"Aston Martin {' '.join(parts[2:]).title()}".strip()
        elif p0 == "land" and p1 == "rover":
            return "Land Rover", " ".join(parts[2:]).title() or "Defender", f"Land Rover {' '.join(parts[2:]).title()}".strip()
        elif p0 == "rolls" and p1 == "royce":
            return "Rolls-Royce", " ".join(parts[2:]).title() or "Phantom", f"Rolls-Royce {' '.join(parts[2:]).title()}".strip()
        elif p0 == "mercedes" and p1 in ["benz", "amg", "maybach"]:
            return "Mercedes-Benz", " ".join(parts[2:]).title() or "C-Class", f"Mercedes-Benz {' '.join(parts[2:]).title()}".strip()
        elif p0 == "force" and p1 == "motors":
            return "Force Motors", " ".join(parts[2:]).title() or "Gurkha", f"Force Motors {' '.join(parts[2:]).title()}".strip()
        else:
            make = parts[0].title()
            model = " ".join(parts[1:]).title()
            return make, model, f"{make} {model}"

    return "Indian Auto", clean.title(), clean.title()


def build_unified_database():
    print("[FUSION] Compiling Canonical Deduplicated Indian Car Catalog...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    canonical_map: Dict[str, Dict[str, Any]] = {}

    def add_image_entry(car_name_str: str, img_path: Path, img_url: str):
        make, model, full_name = standardize_car_name(car_name_str)
        canonical_key = full_name.lower().strip()
        
        if canonical_key not in canonical_map:
            canonical_map[canonical_key] = {
                "id": f"car_{len(canonical_map):04d}",
                "make": make,
                "model": model,
                "full_name": full_name,
                "year_span": "Indian Market Edition",
                "image_filename": img_path.name,
                "image_path": str(img_path),
                "image_url": img_url,
                "exemplars": [str(img_path)]
            }
        else:
            entry = canonical_map[canonical_key]
            if str(img_path) not in entry["exemplars"]:
                entry["exemplars"].append(str(img_path))

    # 1. Source: Indian Car Recommendation System/All car images
    img_dir_1 = Path("Indian Car Recommendation System/All car images")
    if img_dir_1.exists():
        for p in sorted(img_dir_1.glob("*.*")):
            add_image_entry(p.stem, p, f"/dataset_images/{p.name}")

    # 2. Source: Cars Dataset/train (Grouped by class folder name!)
    cars_ds_train = Path("Cars Dataset/train")
    if cars_ds_train.exists():
        for class_dir in sorted(cars_ds_train.iterdir()):
            if class_dir.is_dir():
                class_label = class_dir.name
                imgs = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                for img_p in imgs:
                    add_image_entry(class_label, img_p, f"/cars_dataset_images/{class_dir.name}/{img_p.name}")

    # Re-index unique IDs
    deduped_catalog = list(canonical_map.values())
    for idx, c in enumerate(deduped_catalog):
        c["id"] = f"car_{idx:04d}"
        c["catalog_idx"] = idx

    out_path = Path("data/unified_catalog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped_catalog, f, indent=2)

    total_exemplars = sum(len(c["exemplars"]) for c in deduped_catalog)
    print(f"[FUSION] Successfully compiled {len(deduped_catalog)} UNIQUE canonical Indian car models with {total_exemplars} multi-angle/color exemplars!")
    return deduped_catalog

if __name__ == "__main__":
    build_unified_database()
