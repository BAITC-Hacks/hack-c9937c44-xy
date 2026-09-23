"""CPU smoke tests for inference behavior and checkpoint portability."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from model_agent import ModelAgent, ModelConfig


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        cls.history = rng.normal(size=(4, 6, 4)).astype(np.float32)
        cls.weather = rng.normal(size=(4, 24, 3)).astype(np.float32)
        cls.targets = np.stack(
            [1 / (1 + np.exp(-cls.weather[:, :, 0])), 1 / (1 + np.exp(-cls.weather[:, :, 1]))], axis=-1,
        ).astype(np.float32)
        cls.config = ModelConfig(
            history_features=4, weather_features=3, lookback=6, horizon=24,
            hidden_size=8, num_heads=2, num_layers=1, dropout=0.0,
            epochs=2, batch_size=2, device="cpu",
        )
        cls.agent = ModelAgent(cls.config)
        cls.metrics = cls.agent.fit(cls.history, cls.weather, cls.targets)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_train_predict_and_forecast_sensitivity(self):
        self.assertTrue(np.isfinite(self.metrics["training_mse"]))
        predictions = self.agent.predict(self.history, self.weather)
        self.assertEqual(predictions.shape, (4, 24, 2))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())
        single = self.agent.predict(self.history[0], self.weather[0])
        np.testing.assert_allclose(single, predictions[0], rtol=1e-5, atol=1e-6)
        changed = self.weather[0].copy()
        changed[:, 0] += 3.0
        revised = self.agent.predict(self.history[0], changed)
        self.assertGreater(float(np.max(np.abs(revised - single))), 1e-5)

    def test_checkpoint_restores_weights_and_training_normalization(self):
        reference = self.agent.predict(self.history[0], self.weather[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily" / "model.pt"
            self.agent.save(path)
            restored = ModelAgent.load(path, device="cpu")
            np.testing.assert_allclose(
                restored.predict(self.history[0], self.weather[0]), reference, rtol=1e-6, atol=1e-6,
            )
            self.assertEqual(restored.config, self.config)

    def test_rejects_nonfinite_features_and_invalid_targets(self):
        invalid_weather = self.weather.copy()
        invalid_weather[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.agent.predict(self.history, invalid_weather)
        with self.assertRaisesRegex(ValueError, "NaN"):
            ModelAgent(self.config).fit(self.history, invalid_weather, self.targets)
        with self.assertRaisesRegex(ValueError, "normalized"):
            ModelAgent(self.config).fit(self.history, self.weather, self.targets + 1.0)

    def test_rejects_wrong_horizon_and_untrained_inference(self):
        with self.assertRaisesRegex(ValueError, "weather must have shape"):
            self.agent.predict(self.history, self.weather[:, :-1])
        with self.assertRaisesRegex(RuntimeError, "fit"):
            ModelAgent(self.config).predict(self.history, self.weather)
        with self.assertRaisesRegex(ValueError, "24 or 48"):
            ModelConfig(history_features=4, weather_features=3, horizon=12)

    @unittest.skipIf(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA is available")
    def test_cuda_request_does_not_silently_fall_back_to_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            ModelAgent(ModelConfig(history_features=4, weather_features=3))


if __name__ == "__main__":
    unittest.main()
