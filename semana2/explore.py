import os

import matplotlib

# Uso o backend Agg porque so quero gravar as figuras em ficheiro, nao mostrar janelas.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import medmnist
from medmnist import INFO


RANDOM_STATE = 42
FIGURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")


def load_dataset(flag):
    # Carrego os 3 splits (treino/validacao/teste) e leio os nomes das classes do INFO.
    # Assim o mapeamento 0/1/... vem do dataset e nao e adivinhado por mim.
    info = INFO[flag]
    data_class = getattr(medmnist, info["python_class"])

    splits = {}
    for split in ("train", "val", "test"):
        dataset = data_class(split=split, download=True)
        # imgs: (N, 28, 28) em grayscale ou (N, 28, 28, 3) em RGB. labels: (N, 1).
        splits[split] = (dataset.imgs, dataset.labels.flatten())

    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    n_channels = info["n_channels"]
    return splits, class_names, n_channels


def combine_splits(splits):
    # Junto os 3 splits para olhar para o dataset como um todo.
    imgs = np.concatenate([splits[name][0] for name in splits])
    labels = np.concatenate([splits[name][1] for name in splits])
    return imgs, labels


def class_distribution(flag, splits, class_names):
    # Conto amostras por classe em cada split e no total.
    n_classes = len(class_names)
    counts_by_split = {}
    for split, (_, labels) in splits.items():
        counts_by_split[split] = np.bincount(labels, minlength=n_classes)

    total = np.sum(list(counts_by_split.values()), axis=0)

    # Grafico de barras: numero total de amostras por classe.
    plt.figure(figsize=(max(5, n_classes * 0.9), 4))
    plt.bar(range(n_classes), total)
    plt.xticks(range(n_classes), class_names, rotation=30, ha="right", fontsize=8)
    plt.ylabel("Numero de amostras")
    plt.title(f"Distribuicao das classes - {flag}")
    plt.tight_layout()
    _save(f"{flag}_distribuicao.png")

    return counts_by_split, total


def sample_grid(flag, imgs, labels, class_names, n_channels, k=5):
    # Mostro k exemplos por classe para ver se as classes sao visivelmente diferentes.
    rng = np.random.default_rng(RANDOM_STATE)
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_classes, k, figsize=(k * 1.6, n_classes * 1.6))
    axes = np.atleast_2d(axes)

    for row in range(n_classes):
        idx = np.where(labels == row)[0]
        chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
        for col in range(k):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(chosen):
                img = imgs[chosen[col]]
                if n_channels == 1:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(img)
            if col == 0:
                ax.set_ylabel(class_names[row], fontsize=8)

    fig.suptitle(f"Exemplos por classe - {flag}")
    fig.tight_layout()
    _save(f"{flag}_grelha.png")


def pixel_stats(flag, imgs, labels, class_names, draw_histogram=True):
    # Estatisticas basicas dos pixeis: intervalo, media e desvio (global e por classe).
    stats = {
        "min": float(imgs.min()),
        "max": float(imgs.max()),
        "mean": float(imgs.mean()),
        "std": float(imgs.std()),
    }

    per_class = {}
    for cls, name in enumerate(class_names):
        sub = imgs[labels == cls]
        per_class[name] = (float(sub.mean()), float(sub.std()))

    if draw_histogram:
        # Histograma das intensidades para ver como os pixeis se distribuem.
        plt.figure(figsize=(6, 4))
        plt.hist(imgs.ravel(), bins=50, color="steelblue")
        plt.xlabel("Intensidade do pixel (0-255)")
        plt.ylabel("Frequencia")
        plt.title(f"Histograma das intensidades - {flag}")
        plt.tight_layout()
        _save(f"{flag}_histograma.png")

    return stats, per_class


def mean_image_per_class(flag, imgs, labels, class_names, n_channels):
    # "Imagem media" de cada classe: prova visual de que as classes diferem em media.
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes * 1.8, 2.4))
    axes = np.atleast_1d(axes)

    for cls in range(n_classes):
        media = imgs[labels == cls].astype(np.float64).mean(axis=0)
        ax = axes[cls]
        ax.set_xticks([])
        ax.set_yticks([])
        if n_channels == 1:
            ax.imshow(media, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(media.astype(np.uint8))
        ax.set_title(class_names[cls], fontsize=8)

    fig.suptitle(f"Imagem media por classe - {flag}")
    fig.tight_layout()
    _save(f"{flag}_media_classe.png")


def _save(filename):
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURAS_DIR, filename), dpi=120)
    plt.close()


def explore(flag, draw_histogram=True):
    # Corro a exploracao completa de um dataset e imprimo o resumo no terminal.
    splits, class_names, n_channels = load_dataset(flag)
    imgs, labels = combine_splits(splits)

    counts_by_split, total = class_distribution(flag, splits, class_names)
    sample_grid(flag, imgs, labels, class_names, n_channels)
    stats, per_class = pixel_stats(flag, imgs, labels, class_names, draw_histogram)
    mean_image_per_class(flag, imgs, labels, class_names, n_channels)

    print(f"\n========== {flag} ==========")
    print(f"Canais: {n_channels} | Imagem: {imgs.shape[1:]} | Total amostras: {imgs.shape[0]}")
    print(f"Classes ({len(class_names)}): {class_names}")

    print("\nDistribuicao por split:")
    for split, counts in counts_by_split.items():
        print(f"- {split}: {dict(zip(class_names, counts.tolist()))}")

    print("\nDistribuicao total por classe:")
    soma = int(total.sum())
    for name, count in zip(class_names, total.tolist()):
        print(f"- {name}: {count} ({count / soma:.1%})")

    print("\nEstatisticas dos pixeis (0-255):")
    print(f"- min={stats['min']:.1f} max={stats['max']:.1f} "
          f"media={stats['mean']:.2f} desvio={stats['std']:.2f}")
    print("- media/desvio por classe:")
    for name, (media, desvio) in per_class.items():
        print(f"  - {name}: media={media:.2f} desvio={desvio:.2f}")


def main():
    # BreastMNIST e o dataset principal (continua o cancro da mama da semana1).
    explore("breastmnist", draw_histogram=True)
    # PathMNIST entra como comparacao (RGB, 9 classes).
    explore("pathmnist", draw_histogram=False)

    print(f"\nFiguras gravadas em: {FIGURAS_DIR}")


if __name__ == "__main__":
    main()
