import argparse
import copy
import os
import random
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "semana4-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import BreastMNIST, INFO
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
CLASS_LABELS = [0, 1]
CLASS_NAMES = [INFO["breastmnist"]["label"][str(i)] for i in CLASS_LABELS]


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


class SimpleBreastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_breastmnist_arrays(download=True):
    data = {}
    for split in ("train", "val", "test"):
        dataset = BreastMNIST(split=split, download=download)
        imgs = dataset.imgs.astype(np.float32) / 255.0
        if imgs.ndim == 3:
            imgs = imgs[:, None, :, :]
        else:
            imgs = np.transpose(imgs, (0, 3, 1, 2))
        labels = dataset.labels.flatten().astype(np.int64)
        data[split] = (imgs, labels)
    return data


def make_loader(imgs, labels, batch_size, shuffle):
    x_tensor = torch.tensor(imgs, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(labels, num_classes=2):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / total_items


def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
    )


def train_model(model, loaders, criterion, optimizer, device, config):
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss = evaluate_loss(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoca {epoch:02d}/{config.epochs} "
            f"- train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


def evaluate_model(model, loader, device):
    y_true, y_pred, probabilities = collect_predictions(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=0),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
    }


def predict_from_malignant_probability(probabilities, threshold):
    probabilities = np.asarray(probabilities)
    malignant_probability = probabilities[:, 0]
    return np.where(malignant_probability >= threshold, 0, 1).astype(np.int64)


def evaluate_threshold(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = predict_from_malignant_probability(probabilities, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    true_malignant = cm[0].sum()
    true_benign = cm[1].sum()
    predicted_malignant = cm[:, 0].sum()

    true_positive = cm[0, 0]
    false_benign = cm[0, 1]
    false_malignant = cm[1, 0]
    true_negative = cm[1, 1]

    recall_malignant = true_positive / true_malignant if true_malignant else 0.0
    recall_benign = true_negative / true_benign if true_benign else 0.0
    precision_malignant = (
        true_positive / predicted_malignant if predicted_malignant else 0.0
    )
    if precision_malignant + recall_malignant == 0:
        f2_malignant = 0.0
    else:
        f2_malignant = (
            5
            * precision_malignant
            * recall_malignant
            / (4 * precision_malignant + recall_malignant)
        )

    return {
        "threshold": float(threshold),
        "predictions": y_pred,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": (recall_malignant + recall_benign) / 2,
        "recall_malignant": recall_malignant,
        "recall_benign": recall_benign,
        "precision_malignant": precision_malignant,
        "f2_malignant": f2_malignant,
        "false_benign": int(false_benign),
        "false_malignant": int(false_malignant),
        "confusion_matrix": cm,
    }


def find_best_threshold_by_f2(y_true, probabilities, thresholds):
    results = [
        evaluate_threshold(y_true, probabilities, threshold)
        for threshold in thresholds
    ]
    best_result = max(
        results,
        key=lambda result: (
            result["f2_malignant"],
            result["recall_malignant"],
            result["balanced_accuracy"],
            -abs(result["threshold"] - 0.50),
        ),
    )
    return best_result["threshold"], best_result, results


def prepare_image_tensor(image):
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    array = array.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0

    if array.ndim == 2:
        array = array[None, None, :, :]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.transpose(array, (2, 0, 1))[None, :, :, :]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[None, :, :, :]
    elif array.ndim != 4:
        raise ValueError("A imagem deve ter formato 28x28, 28x28x1 ou 1x28x28.")

    return torch.tensor(array, dtype=torch.float32)


def predict_image(model, image, class_names=CLASS_NAMES, device=None, threshold=None):
    if device is None:
        device = torch.device("cpu")

    model.eval()
    tensor = prepare_image_tensor(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        if threshold is None:
            class_index = int(torch.argmax(probabilities).item())
        else:
            class_index = 0 if float(probabilities[0].item()) >= threshold else 1
        confidence = float(probabilities[class_index].item())

    return {
        "class_index": class_index,
        "class_name": class_names[class_index],
        "confidence": confidence,
        "probabilities": probabilities.cpu().numpy().tolist(),
        "threshold": threshold,
    }


def plot_loss(history):
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], marker="o", label="Treino")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validacao")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Loss da CNN ao longo das epocas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "loss_cnn.png"), dpi=140)
    plt.close()


def plot_confusion_matrix(result):
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=CLASS_NAMES,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("CNN BreastMNIST")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, "confusion_matrix_cnn.png"), dpi=140)
    plt.close(fig)


def plot_wrong_predictions(test_imgs, result, max_images=12):
    wrong_idx = np.where(result["y_true"] != result["y_pred"])[0]
    n_images = min(max_images, len(wrong_idx))
    if n_images == 0:
        return

    os.makedirs(FIGURAS_DIR, exist_ok=True)
    cols = min(4, n_images)
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, wrong_idx[:n_images]):
        image = test_imgs[idx, 0]
        real = CLASS_NAMES[result["y_true"][idx]]
        pred = CLASS_NAMES[result["y_pred"][idx]]
        conf = result["probabilities"][idx, result["y_pred"][idx]]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Real: {real}\nPrevisto: {pred}\nConf.: {conf:.2f}", fontsize=8)

    fig.suptitle("Erros da CNN")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, "erros_cnn.png"), dpi=140)
    plt.close(fig)


def plot_prediction_examples(test_imgs, result, max_each=4):
    correct_idx = np.where(result["y_true"] == result["y_pred"])[0][:max_each]
    wrong_idx = np.where(result["y_true"] != result["y_pred"])[0][:max_each]
    chosen = list(correct_idx) + list(wrong_idx)
    if not chosen:
        return

    os.makedirs(FIGURAS_DIR, exist_ok=True)
    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, chosen):
        image = test_imgs[idx, 0]
        real = CLASS_NAMES[result["y_true"][idx]]
        pred = CLASS_NAMES[result["y_pred"][idx]]
        conf = result["probabilities"][idx, result["y_pred"][idx]]
        estado = "Certo" if result["y_true"][idx] == result["y_pred"][idx] else "Erro"
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{estado}\nReal: {real}\nPrevisto: {pred}\nConf.: {conf:.2f}",
            fontsize=8,
        )

    fig.suptitle("Exemplos de predicoes da CNN")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, "predicoes_cnn.png"), dpi=140)
    plt.close(fig)


def plot_threshold_analysis(threshold_results, best_threshold):
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    thresholds = [result["threshold"] for result in threshold_results]
    recall_malignant = [result["recall_malignant"] for result in threshold_results]
    recall_benign = [result["recall_benign"] for result in threshold_results]

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, recall_malignant, marker="o", label="Recall maligno")
    plt.plot(thresholds, recall_benign, marker="o", label="Recall benigno")
    plt.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        label=f"Limiar escolhido: {best_threshold:.2f}",
    )
    plt.xlabel("Limiar para prever malignant")
    plt.ylabel("Recall")
    plt.title("Analise de limiar da CNN na validacao")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, "threshold_analysis_cnn.png"), dpi=140)
    plt.close()


def print_dataset_summary(data):
    print("Dataset: BreastMNIST")
    for split, (imgs, labels) in data.items():
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        named_counts = dict(zip(CLASS_NAMES, counts.tolist()))
        print(f"- {split}: imagens={imgs.shape}, distribuicao={named_counts}")


def print_results(result):
    errors = int(np.sum(result["y_true"] != result["y_pred"]))
    print("\nResultados no conjunto de teste:")
    print(f"- Accuracy: {result['accuracy']:.4f}")
    print(f"- Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print(f"- Recall maligno: {result['recall_malignant']:.4f}")
    print(f"- Erros: {errors}")
    print("\nConfusion matrix (linhas=classe real, colunas=classe prevista)")
    print("Labels: [malignant, normal/benign]")
    print(result["confusion_matrix"])
    print("\nClassification report:")
    print(result["classification_report"])


def print_threshold_results(best_threshold, val_threshold_result, test_threshold_result):
    print("\nAnalise extra de limiar:")
    print(
        f"- Limiar escolhido na validacao: {best_threshold:.2f} "
        f"(F2 maligno={val_threshold_result['f2_malignant']:.4f}, "
        f"recall maligno={val_threshold_result['recall_malignant']:.4f})"
    )
    print("- Aplicacao desse limiar no teste:")
    print(f"  - Accuracy: {test_threshold_result['accuracy']:.4f}")
    print(f"  - Balanced accuracy: {test_threshold_result['balanced_accuracy']:.4f}")
    print(f"  - Recall maligno: {test_threshold_result['recall_malignant']:.4f}")
    print(f"  - Recall benigno: {test_threshold_result['recall_benign']:.4f}")
    print(f"  - Falsos benignos: {test_threshold_result['false_benign']}")
    print(f"  - Falsos malignos: {test_threshold_result['false_malignant']}")
    print("  - Confusion matrix:")
    print(test_threshold_result["confusion_matrix"])


def run_experiment(config):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    data = load_breastmnist_arrays(download=True)
    print_dataset_summary(data)

    loaders = {
        split: make_loader(imgs, labels, config.batch_size, shuffle=(split == "train"))
        for split, (imgs, labels) in data.items()
    }

    train_labels = data["train"][1]
    class_weights = compute_class_weights(train_labels, num_classes=2).to(device)
    print(f"\nPesos por classe: {class_weights.cpu().numpy().round(4).tolist()}")

    model = SimpleBreastCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history, best_epoch, best_val_loss = train_model(
        model, loaders, criterion, optimizer, device, config
    )
    print(
        f"\nMelhor epoca pela loss de validacao: {best_epoch} "
        f"(val_loss={best_val_loss:.4f})"
    )

    val_result = evaluate_model(model, loaders["val"], device)
    result = evaluate_model(model, loaders["test"], device)
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    best_threshold, val_threshold_result, threshold_results = find_best_threshold_by_f2(
        val_result["y_true"],
        val_result["probabilities"],
        thresholds,
    )
    test_threshold_result = evaluate_threshold(
        result["y_true"],
        result["probabilities"],
        best_threshold,
    )

    plot_loss(history)
    plot_confusion_matrix(result)
    plot_wrong_predictions(data["test"][0], result)
    plot_prediction_examples(data["test"][0], result)
    plot_threshold_analysis(threshold_results, best_threshold)
    print_results(result)
    print_threshold_results(best_threshold, val_threshold_result, test_threshold_result)
    print(f"\nFiguras gravadas em: {FIGURAS_DIR}")

    example_correct = np.where(result["y_true"] == result["y_pred"])[0]
    example_wrong = np.where(result["y_true"] != result["y_pred"])[0]
    if len(example_correct) > 0:
        idx = int(example_correct[0])
        prediction = predict_image(model, data["test"][0][idx], CLASS_NAMES, device)
        print(
            "\nExemplo correto: "
            f"idx={idx}, real={CLASS_NAMES[result['y_true'][idx]]}, "
            f"previsto={prediction['class_name']}, "
            f"confianca={prediction['confidence']:.4f}"
        )
    if len(example_wrong) > 0:
        idx = int(example_wrong[0])
        prediction = predict_image(model, data["test"][0][idx], CLASS_NAMES, device)
        print(
            "Exemplo errado: "
            f"idx={idx}, real={CLASS_NAMES[result['y_true'][idx]]}, "
            f"previsto={prediction['class_name']}, "
            f"confianca={prediction['confidence']:.4f}"
        )

    return history, result, best_epoch, best_val_loss, test_threshold_result


def parse_args():
    parser = argparse.ArgumentParser(description="CNN simples para BreastMNIST.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
