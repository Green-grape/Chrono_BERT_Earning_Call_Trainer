from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chrono_dapt.training import (
    PRE_PRESSURE_CLEANUP_CODE,
    compatible_resume_identity,
    trim_cuda_cache_if_needed,
)


class CachePressureTests(unittest.TestCase):
    def test_cpu_never_touches_cuda(self):
        with patch("chrono_dapt.training.torch.cuda.empty_cache") as empty:
            self.assertFalse(trim_cuda_cache_if_needed(SimpleNamespace(device="cpu"), 2023, "presentation", 25))
            empty.assert_not_called()

    def check_pressure(self, reserved_gib, allocated_gib, expected):
        gib = 1024 ** 3
        with (patch("chrono_dapt.training.torch.cuda.is_available", return_value=True),
              patch("chrono_dapt.training.torch.cuda.memory_reserved", return_value=reserved_gib * gib),
              patch("chrono_dapt.training.torch.cuda.memory_allocated", return_value=allocated_gib * gib),
              patch("chrono_dapt.training.torch.cuda.current_device", return_value=0),
              patch("chrono_dapt.training.torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=16 * gib)),
              patch("chrono_dapt.training.log_cuda_memory") as log,
              patch("chrono_dapt.training.torch.cuda.empty_cache") as empty):
            self.assertEqual(trim_cuda_cache_if_needed(SimpleNamespace(device="cuda"), 2023, "presentation", 25), expected)
            self.assertEqual(empty.call_count, int(expected))
            self.assertEqual(log.call_count, 2 * int(expected))

    def test_cache_below_pressure_threshold_is_kept(self):
        self.check_pressure(9.7, 2.4, False)

    def test_large_idle_cache_is_trimmed(self):
        self.check_pressure(12, 2.4, True)

    def test_live_tensor_pressure_is_not_treated_as_idle_cache(self):
        self.check_pressure(12, 11.5, False)

    def test_previous_diagnostic_code_can_resume_but_unknown_code_cannot(self):
        current = {"code": {"training.py": "current"}, "settings": {"seed": 42}}
        previous = {**current, "code": dict(PRE_PRESSURE_CLEANUP_CODE)}
        self.assertTrue(compatible_resume_identity(previous, current))
        unknown = {**current, "code": {"training.py": "unknown"}}
        self.assertFalse(compatible_resume_identity(unknown, current))
