from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


RANDOM_STATE = 42


def load_data():
    # Carrego o dataset e transformo em DataFrame para ser mais facil de ler.
    data = load_breast_cancer()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    target_names = [str(name) for name in data.target_names]
    return data, x, y, target_names


def build_model():
    # O scaler normaliza as variaveis antes da regressao logistica.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def get_class_distribution(y, target_names):
    counts = y.value_counts().sort_index()
    return {target_names[index]: int(counts[index]) for index in counts.index}


def run_baseline(show_plot=False):
    data, x, y, target_names = load_data()

    # O stratify mantem a proporcao entre benigno e maligno no treino/teste.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    # Primeiro o modelo aprende com o treino, depois prevê exemplos do teste.
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if show_plot:
        plot_class_distribution(y, target_names)

    return {
        "data": data,
        "features": x,
        "target": y,
        "target_names": target_names,
        "dataset_shape": x.shape,
        "train_shape": x_train.shape,
        "test_shape": x_test.shape,
        "class_distribution": get_class_distribution(y, target_names),
        "feature_summary": x.describe(),
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        # Labels [0, 1] significam [malignant, benign].
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=target_names,
        ),
    }


def plot_class_distribution(y, target_names):
    import matplotlib.pyplot as plt

    labels = [target_names[value] for value in y]
    pd.Series(labels).value_counts().sort_index().plot(kind="bar")
    plt.title("Distribuicao das classes")
    plt.xlabel("Classe")
    plt.ylabel("Numero de amostras")
    plt.tight_layout()
    plt.show()


def main():
    results = run_baseline(show_plot=False)

    print("Dataset: Breast Cancer Wisconsin Diagnostic")
    print(f"Shape X: {results['dataset_shape']}")
    print(f"Shape y: {results['target'].shape}")
    print(f"Classes: {results['target_names']} (0=malignant, 1=benign)")
    print("\nPrimeiras linhas:")
    print(results["features"].head())
    print("\nDistribuicao das classes:")
    for label, count in results["class_distribution"].items():
        print(f"- {label}: {count}")
    print("\nResumo das variaveis:")
    print(results["feature_summary"])
    print(f"\nSplit treino/teste: {results['train_shape']} / {results['test_shape']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nConfusion matrix (linhas=classe real, colunas=classe prevista)")
    print("Labels: [malignant, benign]")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(results["classification_report"])


if __name__ == "__main__":
    main()
