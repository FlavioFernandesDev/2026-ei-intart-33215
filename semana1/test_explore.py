import unittest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semana1 import explore


class Semana1BaselineTest(unittest.TestCase):
    def test_baseline_uses_expected_model_and_metrics(self):
        results = explore.run_baseline(show_plot=False)

        self.assertEqual(results["dataset_shape"], (569, 30))
        self.assertEqual(results["target_names"], ["malignant", "benign"])
        self.assertEqual(results["class_distribution"], {"malignant": 212, "benign": 357})
        self.assertEqual(results["train_shape"], (455, 30))
        self.assertEqual(results["test_shape"], (114, 30))
        self.assertAlmostEqual(results["accuracy"], 0.9824561403508771)
        self.assertEqual(results["confusion_matrix"].tolist(), [[41, 1], [1, 71]])

        model = results["model"]
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(model.named_steps["classifier"], LogisticRegression)
        self.assertEqual(model.named_steps["classifier"].random_state, 42)


if __name__ == "__main__":
    unittest.main()
