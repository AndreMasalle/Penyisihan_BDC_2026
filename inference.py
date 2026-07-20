"""
Inference Real-Time dengan Webcam
=================================
Klasifikasi sampah (Electronic / Organic / Recyclable)
menggunakan model Swin Transformer Tiny yang sudah di-train.

Cara pakai:
    python inference.py

Tekan 'q' untuk keluar dari jendela kamera.
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path


# ============================================================
# Konfigurasi
# ============================================================
MODEL_PATH = Path(__file__).parent / "swin_t.pth"
NUM_CLASSES = 3
CLASS_NAMES = ["Electronic", "Organic", "Recyclable"]

# Warna untuk masing-masing kelas (BGR untuk OpenCV)
CLASS_COLORS = {
    "Electronic": (255, 165, 0),   # Oranye
    "Organic":    (0, 200, 0),     # Hijau
    "Recyclable": (0, 180, 255),   # Biru muda
}

# Transform yang sama dengan eval transform saat training
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),           # numpy (H,W,C) -> PIL Image
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Fungsi pembuatan model (sama dengan notebook)
# ============================================================
def build_model(model_name: str = "swin_t", num_classes: int = 3):
    """
    Factory model — harus sama dengan arsitektur yang dipakai saat training.
    """
    match model_name:
        case "resnet50":
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        case "efficientnet_b3":
            m = models.efficientnet_b3(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        case "convnext_tiny":
            m = models.convnext_tiny(weights=None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        case "swin_t":
            m = models.swin_t(weights=None)
            m.head = nn.Linear(m.head.in_features, num_classes)
        case _:
            raise ValueError(f"Model '{model_name}' tidak dikenali.")
    return m


# ============================================================
# Load model
# ============================================================
def load_model(model_path: Path, device: torch.device):
    """
    Membangun arsitektur Swin-T lalu load state_dict dari file .pth.
    """
    model = build_model("swin_t", NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Model berhasil di-load dari: {model_path}")
    print(f"[INFO] Device: {device}")
    return model


# ============================================================
# Prediksi satu frame
# ============================================================
@torch.no_grad()
def predict_frame(frame_rgb: np.ndarray, model, device: torch.device):
    """
    Menerima frame BGR dari OpenCV, lalu mengembalikan:
        - predicted_class (str)
        - confidence (float, 0-1)
    """
    # Transform: numpy (H,W,C) RGB -> tensor (1,C,H,W)
    tensor = EVAL_TRANSFORM(frame_rgb).unsqueeze(0).to(device)

    # Forward pass
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    confidence, idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[idx.item()]
    confidence_value = confidence.item()

    return predicted_class, confidence_value


# ============================================================
# Gambar overlay di frame
# ============================================================
def draw_overlay(frame, predicted_class: str, confidence: float, fps: float):
    """
    Menggambar label prediksi, confidence, dan FPS di atas frame.
    """
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(predicted_class, (255, 255, 255))

    # ---- Background semi-transparan di bagian atas ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ---- Label prediksi ----
    label_text = f"{predicted_class}"
    cv2.putText(
        frame, label_text, (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
    )

    # ---- Confidence ----
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(
        frame, conf_text, (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
    )

    # ---- FPS di pojok kanan atas ----
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(
        frame, fps_text, (w - tw - 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
    )

    # ---- Garis bawah berwarna sesuai kelas ----
    cv2.rectangle(frame, (0, 97), (w, 100), color, -1)

    return frame


# ============================================================
# Main — loop kamera
# ============================================================
def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(MODEL_PATH, device)

    # Buka webcam (index 0 = kamera default)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka kamera. Pastikan webcam terhubung.")
        return

    print("[INFO] Kamera terbuka. Tekan 'q' untuk keluar.")

    fps = 0.0
    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal membaca frame, mencoba lagi...")
            continue

        # Hitung FPS
        curr_time = cv2.getTickCount()
        time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # OpenCV membaca dalam BGR, konversi ke RGB untuk model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prediksi
        predicted_class, confidence = predict_frame(frame_rgb, model, device)

        # Gambar overlay
        frame = draw_overlay(frame, predicted_class, confidence, fps)

        # Tampilkan
        cv2.imshow("Klasifikasi Sampah - Swin-T (tekan 'q' untuk keluar)", frame)

        # Tekan 'q' untuk keluar
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Kamera ditutup. Selesai.")


if __name__ == "__main__":
    main()
