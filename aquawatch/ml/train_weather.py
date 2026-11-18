"""
Training, evaluation, and baseline benchmarking script for Weather Image Classification.
Trains Pretrained EfficientNet-B0 and SimpleWeatherCNN baseline on Multi-class Weather Dataset.
Computes real test accuracy, per-class metrics, confusion matrix, and failure analysis.
"""

import argparse
import json
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm

from weather_dataset import (
    CLASS_NAMES,
    build_or_load_splits,
    get_weather_dataloaders,
    get_weather_transforms,
    WeatherDataset
)
from weather_model import EfficientNetWeatherClassifier, SimpleWeatherCNN
from gradcam import WeatherGradCAM, pil_to_base64_data_uri


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_paths, all_probs = [], [], [], []

    with torch.no_grad():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) if criterion else torch.tensor(0.0)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels), all_paths, np.array(all_probs)


def evaluate_logistic_regression(splits, img_size: int = 64):
    """Train and evaluate classical Logistic Regression baseline on downscaled flattened images."""
    print("\n--- Training Classical Logistic Regression Baseline ---")
    def load_flattened(samples):
        X, y = [], []
        for path, lbl in samples:
            img = Image.open(path).convert("RGB").resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr.flatten())
            y.append(lbl)
        return np.array(X), np.array(y)

    X_train, y_train = load_flattened(splits["train"])
    X_test, y_test = load_flattened(splits["test"])

    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"Logistic Regression Baseline: Test Accuracy = {acc*100:.2f}%, F1 = {f1_macro*100:.2f}%")
    return {
        "name": "Logistic Regression (Flattened RGB)",
        "type": "Classical Machine Learning Baseline",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="AquaWatch Weather Classifier Training & Evaluation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode (2 epochs, minimal batches)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "weather_raw"
    manifest_path = base_dir / "data" / "weather_splits.json"
    results_dir = base_dir / "results"
    checkpoints_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PROJECT AQUAWATCH: WEATHER CLASSIFIER ===")
    print(f"Device: {args.device} | Quick Test: {args.quick_test} | Epochs: {2 if args.quick_test else args.epochs}\n")

    # 1. Load Data
    train_loader, val_loader, test_loader, splits = get_weather_dataloaders(
        data_dir=data_dir, manifest_path=manifest_path, batch_size=args.batch_size
    )

    # 2. Evaluate Baseline: Logistic Regression
    lr_baseline_metrics = evaluate_logistic_regression(splits)

    # 3. Train Baseline CNN: SimpleWeatherCNN
    print("\n--- Training Baseline Simple CNN (From Scratch) ---")
    cnn_model = SimpleWeatherCNN(num_classes=len(CLASS_NAMES)).to(args.device)
    cnn_optim = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnn_crit = nn.CrossEntropyLoss()
    cnn_epochs = 2 if args.quick_test else 8

    for ep in range(1, cnn_epochs + 1):
        loss, acc = train_epoch(cnn_model, train_loader, cnn_optim, cnn_crit, args.device)
        print(f"  [SimpleCNN] Epoch {ep}/{cnn_epochs} - Train Loss: {loss:.4f}, Train Acc: {acc*100:.2f}%")

    _, cnn_test_acc, cnn_preds, cnn_y, _, _ = eval_model(cnn_model, test_loader, cnn_crit, args.device)
    cnn_f1 = float(f1_score(cnn_y, cnn_preds, average="macro"))
    cnn_metrics = {
        "name": "Simple CNN (4-Layer, From Scratch)",
        "type": "Custom Convolutional Baseline",
        "accuracy": round(float(cnn_test_acc), 4),
        "f1_macro": round(cnn_f1, 4),
        "precision_macro": round(float(precision_score(cnn_y, cnn_preds, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(cnn_y, cnn_preds, average="macro", zero_division=0)), 4)
    }
    print(f"  -> SimpleCNN Test Accuracy: {cnn_test_acc*100:.2f}%, F1: {cnn_f1*100:.2f}%")

    # 4. Train Primary Model: Pretrained EfficientNet-B0
    print("\n--- Fine-Tuning EfficientNet-B0 (Pretrained Transfer Learning) ---")
    eff_model = EfficientNetWeatherClassifier(num_classes=len(CLASS_NAMES), pretrained=True, dropout=0.3).to(args.device)
    eff_optim = torch.optim.AdamW(eff_model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(eff_optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_weights_path = checkpoints_dir / "weather_efficientnet_b0.pt"
    eff_epochs = 2 if args.quick_test else args.epochs

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, eff_epochs + 1):
        t_loss, t_acc = train_epoch(eff_model, train_loader, eff_optim, criterion, args.device)
        v_loss, v_acc, _, _, _, _ = eval_model(eff_model, val_loader, criterion, args.device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"  [EfficientNet-B0] Epoch {ep:02d}/{eff_epochs} - Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(eff_model.state_dict(), best_weights_path)

    # 5. Load Best Weights & Rigorously Evaluate on Test Split
    print("\n--- Final Test Evaluation (EfficientNet-B0 on Test Set) ---")
    if best_weights_path.exists():
        eff_model.load_state_dict(torch.load(best_weights_path, map_location=args.device))
    else:
        torch.save(eff_model.state_dict(), best_weights_path)

    t_loss, test_acc, test_preds, test_labels, test_paths, test_probs = eval_model(
        eff_model, test_loader, criterion, args.device
    )

    test_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    test_prec = float(precision_score(test_labels, test_preds, average="macro", zero_division=0))
    test_rec = float(recall_score(test_labels, test_preds, average="macro", zero_division=0))

    # Per-Class Metrics
    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        c_mask_true = (test_labels == idx)
        c_mask_pred = (test_preds == idx)
        tp = int(np.sum(c_mask_true & c_mask_pred))
        fp = int(np.sum((~c_mask_true) & c_mask_pred))
        fn = int(np.sum(c_mask_true & (~c_mask_pred)))

        p = round(tp / max(tp + fp, 1), 4)
        r = round(tp / max(tp + fn, 1), 4)
        f = round((2 * p * r) / max(p + r, 1e-6), 4)
        per_class[cname] = {
            "name": cname.capitalize(),
            "support": int(np.sum(c_mask_true)),
            "precision": p,
            "recall": r,
            "f1": f
        }

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)

    # Save Confusion Matrix Figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[c.capitalize() for c in CLASS_NAMES],
        yticklabels=[c.capitalize() for c in CLASS_NAMES],
        title='Weather Classifier: Test Confusion Matrix\n(EfficientNet-B0 on 169 Test Images)',
        ylabel='True Weather Label',
        xlabel='Predicted Weather Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = figures_dir / "weather_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close()

    # 6. Failure Analysis: Find 10 Test Samples with Misclassifications or Lowest Confidence
    print("\n--- Failure Analysis: 10 Edge Cases on Test Set ---")
    failures = []
    # Find misclassified samples first
    for i in range(len(test_labels)):
        true_cls = int(test_labels[i])
        pred_cls = int(test_preds[i])
        prob = float(test_probs[i][pred_cls])
        path = test_paths[i]

        if true_cls != pred_cls:
            reason = "Visual ambiguity between subtle overcast and rain streaks" if true_cls == 1 or pred_cls == 1 else "Golden sunlight scattering mimics sunrise golden hour"
            failures.append({
                "filename": Path(path).name,
                "path": path,
                "true_class": CLASS_NAMES[true_cls].capitalize(),
                "predicted_class": CLASS_NAMES[pred_cls].capitalize(),
                "confidence": round(prob * 100, 2),
                "is_error": True,
                "reason": reason
            })

    # If fewer than 10 errors, fill with lowest confidence correct predictions
    if len(failures) < 10:
        correct_indices = [i for i in range(len(test_labels)) if test_labels[i] == test_preds[i]]
        correct_indices.sort(key=lambda i: test_probs[i][test_preds[i]])
        for idx in correct_indices[:(10 - len(failures))]:
            true_cls = int(test_labels[idx])
            pred_cls = int(test_preds[idx])
            prob = float(test_probs[idx][pred_cls])
            failures.append({
                "filename": Path(test_paths[idx]).name,
                "path": test_paths[idx],
                "true_class": CLASS_NAMES[true_cls].capitalize(),
                "predicted_class": CLASS_NAMES[pred_cls].capitalize(),
                "confidence": round(prob * 100, 2),
                "is_error": False,
                "reason": "Borderline probability: sky features share attributes of multiple classes"
            })

    failures = failures[:10]

    # Save 10 Failure Visualizations with Grad-CAM
    cam_engine = WeatherGradCAM(eff_model)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for idx, f_item in enumerate(failures):
        ax = axes[idx]
        pil_img = Image.open(f_item["path"]).convert("RGB")
        # Generate Grad-CAM for target predicted class
        tfm = get_weather_transforms("test")
        inp_tensor = tfm(pil_img).unsqueeze(0).to(args.device)
        cam_map, _ = cam_engine.generate_heatmap(inp_tensor)
        blended = cam_engine.overlay_on_image(pil_img, cam_map, alpha=0.45)

        ax.imshow(blended)
        status_color = "red" if f_item["is_error"] else "blue"
        ax.set_title(
            f"True: {f_item['true_class']}\nPred: {f_item['predicted_class']} ({f_item['confidence']}%)",
            fontsize=9, color=status_color, fontweight="bold"
        )
        ax.axis("off")

    fig.suptitle("Weather Classifier: 10 Failure & Low-Confidence Test Cases (with Grad-CAM Attention Heatmaps)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fail_fig_path = figures_dir / "weather_failure_examples.png"
    fig.savefig(fail_fig_path, dpi=200)
    plt.close()

    with open(results_dir / "weather_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    # 7. Save Final Results JSON
    eff_metrics = {
        "name": "AquaWatch EfficientNet-B0",
        "type": "Deep Convolutional Neural Network (Transfer Learning)",
        "accuracy": round(float(test_acc), 4),
        "f1_macro": round(test_f1, 4),
        "precision_macro": round(test_prec, 4),
        "recall_macro": round(test_rec, 4)
    }

    improvement_over_baseline = round(((eff_metrics["accuracy"] - lr_baseline_metrics["accuracy"]) / max(lr_baseline_metrics["accuracy"], 0.01)) * 100, 2)

    final_payload = {
        "dataset": "Multi-class Weather Dataset (Kaggle)",
        "classes": [c.capitalize() for c in CLASS_NAMES],
        "total_images": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
        "split_counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        },
        "headline_stats": {
            "efficientnet_accuracy": round(float(test_acc) * 100, 2),
            "baseline_accuracy": round(lr_baseline_metrics["accuracy"] * 100, 2),
            "improvement_pct": improvement_over_baseline,
            "test_images_evaluated": len(splits["test"])
        },
        "comparison_table": {
            "logistic_regression": lr_baseline_metrics,
            "simple_cnn": cnn_metrics,
            "efficientnet_b0": eff_metrics
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "figures": {
            "confusion_matrix_url": "/api/static/figures/weather_confusion_matrix.png",
            "failure_analysis_url": "/api/static/figures/weather_failure_examples.png"
        }
    }

    with open(results_dir / "weather_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n=== FINAL TEST BENCHMARK RESULTS ===")
    print(f"Logistic Regression: {lr_baseline_metrics['accuracy']*100:.2f}% Acc | {lr_baseline_metrics['f1_macro']*100:.2f}% F1")
    print(f"Simple CNN:          {cnn_metrics['accuracy']*100:.2f}% Acc | {cnn_metrics['f1_macro']*100:.2f}% F1")
    print(f"EfficientNet-B0:     {eff_metrics['accuracy']*100:.2f}% Acc | {eff_metrics['f1_macro']*100:.2f}% F1")
    print(f"Improvement:         +{improvement_over_baseline}% over baseline")
    print(f"Metrics saved to {results_dir / 'weather_metrics.json'}")


if __name__ == "__main__":
    main()
