# 🏎️ Car Detection & Fine-Grained Classification: Machine Learning Student Guide

Welcome to the **Indian Car Make, Model & Year Detection System**! This guide is designed to explain the core machine learning concepts, mathematics, and architecture decisions implemented in this project.

---

## 1. Problem Formulation: Fine-Grained Visual Categorization (FGVC)

In standard image classification (e.g. ImageNet), categories are distinct (e.g. Dog vs. Airplane vs. Coffee Mug). The network relies on macro-features: wings, fur, wheels.

In **Fine-Grained Classification** (e.g. distinguishing a *2018 Maruti Swift* from a *2023 Maruti Swift* or *Tata Nexon*):
- All classes share the same global geometry (4 wheels, windshield, roof, doors).
- Differentiating features are concentrated in localized, subtle design cues:
  1. **Front Grille**: Hexagonal mesh (Swift), Cascading jewel (Creta), 6-slat retro (Thar).
  2. **Lighting Signatures**: Split DRLs (Nexon), Z-shaped LEDs (i20), C-shaped LED brackets (XUV700).
  3. **C-Pillar & Beltline**: Floating roof lines, kick-up quarter glass, hidden handles.

---

## 2. Transfer Learning: Why Pretrained Weights Matter

Training deep vision networks from scratch requires millions of images. Instead, we use **Transfer Learning**:
1. A backbone network (e.g., **EfficientNet-B0** or **ResNet-34**) is already trained on ImageNet (1.4 million images).
2. Its early layers have learned universal visual primitives (edges, textures, gradients, curves).
3. Its middle/deep layers detect object parts (wheels, lights, contours).
4. We freeze/fine-tune the backbone and replace the final Linear classification head:
   $$\text{Head}(x) = \text{Linear}(\text{in\_features}=1280 \to \text{out\_features}=12)$$

---

## 3. The Math Behind the Loss & Optimization

### Cross-Entropy Loss with Label Smoothing
For true class $y$ and predicted probabilities $p_i = \text{softmax}(z_i)$:
$$\mathcal{L}_{CE} = -\sum_{i=1}^C q_i \log(p_i)$$

Where standard one-hot encoding $q_y = 1$ is softened with smoothing factor $\epsilon = 0.1$:
$$q_i = \begin{cases} 1 - \epsilon + \frac{\epsilon}{C} & \text{if } i = y \\ \frac{\epsilon}{C} & \text{if } i \neq y \end{cases}$$
**Why?** In fine-grained car classification, label smoothing prevents over-confidence and acts as a strong regularizer.

### AdamW Optimizer with Weight Decay
$$\theta_{t+1} = \theta_t - \gamma \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
Decouples weight decay ($\lambda$) from gradient updates, preventing weight explosion and improving generalization.

---

## 4. Hardware Optimization for RTX 3050 (4GB VRAM)

Your laptop's **NVIDIA RTX 3050** has Tensor Cores and 4GB of high-speed GDDR6 VRAM.
We enable **Automatic Mixed Precision (AMP / FP16)**:
- Performs matrix multiplications in 16-bit floating point (half precision).
- Reduces memory usage by ~50% and doubles throughput.
- Uses a `GradScaler` to prevent gradient underflow when backpropagating small gradient values:
```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast(device_type="cuda"):
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 5. Explainability: How Grad-CAM Works

**Grad-CAM (Gradient-weighted Class Activation Mapping)** reveals what spatial regions of the image the neural network focused on to make its decision.

1. Compute gradient of score for class $c$ ($y^c$) with respect to feature activation map $A^k$ of the last conv layer:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
2. Compute weighted combination followed by ReLU (to only highlight positive influences):
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
3. Upsample the resulting 2D heatmap and overlay onto the car image.

---

## 6. How to Run & Expand the Project

### Running the Web Application:
```powershell
python app.py
```
Open your browser at `http://127.0.0.1:8000` to interact with the Black & White Retro-Gaming UI!

### Training on RTX 3050:
- Use the web UI's **[ 3. GPU TRAINER ]** tab, or run in terminal:
```powershell
python src/train.py
```

### Adding New Car Classes:
1. Add the car's metadata to `src/indian_cars_metadata.py`.
2. Add reference or scraped images to `data/train/<class_id>/` and `data/val/<class_id>/`.
3. Re-run training!
