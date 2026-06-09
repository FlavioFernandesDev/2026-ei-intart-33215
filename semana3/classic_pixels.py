import os
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "semana3-matplotlib-cache"),
)

import matplotlib

# Uso o backend Agg porque so quero gravar as figuras em ficheiro.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from medmnist import BreastMNIST, INFO
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from skimage.feature import hog
except ImportError:
    hog = None


RANDOM_STATE = 42
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")
CLASS_LABELS = [0, 1]


def load_breastmnist():
    info = INFO["breastmnist"]
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

    data = {}
    for split in ("train", "val", "test"):
        dataset = BreastMNIST(split=split, download=True)
        imgs = dataset.imgs.astype(np.float32) / 255.0
        labels = dataset.labels.flatten().astype(int)
        data[split] = (imgs, labels)

    return data, class_names


def flatten_images(imgs):
    # Cada imagem 28x28 passa a ser um vetor com 784 valores.
    return imgs.reshape(imgs.shape[0], -1)


def extract_hog_features(imgs):
    if hog is None:
        raise RuntimeError(
            "scikit-image nao esta instalado. Instala com: pip install scikit-image"
        )

    features = []
    for img in imgs:
        features.append(
            hog(
                img,
                orientations=9,
                pixels_per_cell=(7, 7),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.asarray(features, dtype=np.float32)


def build_models():
    return {
        "LogisticRegression pixels": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVC RBF pixels": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", class_weight="balanced")),
            ]
        ),
    }


def evaluate_model(name, model, x_test, y_test, class_names):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    return {
        "name": name,
        "predictions": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        # No BreastMNIST a classe 0 corresponde a malignant.
        "recall_malignant": recall_score(y_test, y_pred, pos_label=0),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=CLASS_LABELS,
            target_names=class_names,
            zero_division=0,
        ),
    }


def plot_confusion_matrix(result, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(result["name"])
    fig.tight_layout()
    _save(f"confusion_matrix_{slug(result['name'])}.png")


def plot_wrong_predictions(result, test_imgs, y_test, class_names, max_images=12):
    wrong_idx = np.where(result["predictions"] != y_test)[0]
    n_images = min(max_images, len(wrong_idx))
    if n_images == 0:
        return

    cols = min(4, n_images)
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, wrong_idx[:n_images]):
        ax.imshow(test_imgs[idx], cmap="gray", vmin=0, vmax=1)
        real = class_names[y_test[idx]]
        pred = class_names[result["predictions"][idx]]
        ax.set_title(f"Real: {real}\nPrevisto: {pred}", fontsize=8)

    fig.suptitle(f"Erros - {result['name']}")
    fig.tight_layout()
    _save(f"erros_{slug(result['name'])}.png")


def plot_model_comparison(results):
    names = [result["name"] for result in results]
    accuracy = [result["accuracy"] for result in results]
    balanced = [result["balanced_accuracy"] for result in results]
    recall_malignant = [result["recall_malignant"] for result in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, accuracy, width, label="Accuracy")
    ax.bar(x, balanced, width, label="Balanced accuracy")
    ax.bar(x + width, recall_malignant, width, label="Recall malignant")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacao dos modelos")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    _save("comparacao_modelos.png")


def _save(filename):
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURAS_DIR, filename), dpi=140)
    plt.close()


def slug(value):
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def print_dataset_summary(data, class_names):
    print("Dataset: BreastMNIST")
    for split, (imgs, labels) in data.items():
        counts = np.bincount(labels, minlength=len(class_names))
        named_counts = dict(zip(class_names, counts.tolist()))
        print(f"- {split}: imagens={imgs.shape}, distribuicao={named_counts}")


def print_results(results, y_test):
    print("\nResumo dos resultados no conjunto de teste:")
    print(
        f"{'Modelo':<32} {'Accuracy':>9} {'Bal.Acc':>9} "
        f"{'Recall maligno':>15} {'Erros':>8}"
    )
    for result in results:
        errors = int(np.sum(result["predictions"] != y_test))
        print(
            f"{result['name']:<32} "
            f"{result['accuracy']:>9.4f} "
            f"{result['balanced_accuracy']:>9.4f} "
            f"{result['recall_malignant']:>15.4f} "
            f"{errors:>8}"
        )

    for result in results:
        print(f"\n=== {result['name']} ===")
        print("Confusion matrix (linhas=classe real, colunas=classe prevista)")
        print(result["confusion_matrix"])
        print("\nClassification report:")
        print(result["classification_report"])


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data, class_names = load_breastmnist()
    print_dataset_summary(data, class_names)

    train_imgs, y_train = data["train"]
    val_imgs, y_val = data["val"]
    test_imgs, y_test = data["test"]

    x_train = flatten_images(train_imgs)
    x_val = flatten_images(val_imgs)
    x_test = flatten_images(test_imgs)

    results = []
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        print(
            f"\nValidacao - {name}: "
            f"balanced_accuracy={balanced_accuracy_score(y_val, val_pred):.4f}"
        )
        result = evaluate_model(name, model, x_test, y_test, class_names)
        results.append(result)
        plot_confusion_matrix(result, class_names)
        plot_wrong_predictions(result, test_imgs, y_test, class_names)

    if hog is not None:
        x_train_hog = extract_hog_features(train_imgs)
        x_val_hog = extract_hog_features(val_imgs)
        x_test_hog = extract_hog_features(test_imgs)
        hog_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        hog_model.fit(x_train_hog, y_train)
        val_pred = hog_model.predict(x_val_hog)
        print(
            "\nValidacao - LogisticRegression HOG: "
            f"balanced_accuracy={balanced_accuracy_score(y_val, val_pred):.4f}"
        )
        result = evaluate_model(
            "LogisticRegression HOG", hog_model, x_test_hog, y_test, class_names
        )
        results.append(result)
        plot_confusion_matrix(result, class_names)
        plot_wrong_predictions(result, test_imgs, y_test, class_names)
    else:
        print("\nHOG ignorado: scikit-image nao esta instalado.")

    plot_model_comparison(results)

    print_results(results, y_test)
    print(f"\nFiguras gravadas em: {FIGURAS_DIR}")


if __name__ == "__main__":
    main()
