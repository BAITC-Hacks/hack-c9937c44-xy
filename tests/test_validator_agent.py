import unittest

import numpy as np

from validator_agent import PhysicsLimits, validate_power


class PhysicsValidatorTests(unittest.TestCase):
    def test_physical_boundaries_and_clipping(self):
        power = np.array([[-0.2, 1.2], [0.7, 0.6], [0.4, 0.8]])
        wind = np.array([[10.0, 10.0], [2.99, 25.01], [3.0, 25.0]])
        result = validate_power(power, wind)
        np.testing.assert_allclose(result, [[0, 1], [0, 0], [0.4, 0.8]])
        self.assertEqual(power[0, 0], -0.2, "Validation must not mutate model output.")

    def test_10m_wind_does_not_force_hub_height_shutdown(self):
        power = np.array([[0.7, 1.2]])
        wind = np.array([[2.0, 26.0]])
        np.testing.assert_allclose(
            validate_power(power, wind, apply_wind_limits=False), [[0.7, 1.0]],
        )

    def test_nonfinite_or_negative_wind_fails(self):
        for power, wind in [
            ([[np.nan]], [[8]]), ([[np.inf]], [[8]]),
            ([[0.5]], [[np.nan]]), ([[0.5]], [[-1]]),
        ]:
            with self.subTest(power=power, wind=wind), self.assertRaises(ValueError):
                validate_power(np.array(power), np.array(wind))

    def test_invalid_shapes_and_limits_fail(self):
        for power, wind in [
            (np.ones(3), np.ones(3)), (np.ones((2, 2)), np.ones((2, 1))),
            (np.empty((0, 2)), np.empty((0, 2))),
        ]:
            with self.subTest(shape=power.shape), self.assertRaises(ValueError):
                validate_power(power, wind)
        for limits in [(-1, 25), (25, 3), (3, np.inf)]:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                PhysicsLimits(*limits)


if __name__ == "__main__":
    unittest.main()
