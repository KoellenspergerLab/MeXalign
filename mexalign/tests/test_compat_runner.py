import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mexalign.compat_runner import (
    _set_cfg_key,
    build_output_paths,
    resolve_eval_reference,
)
from mexalign.core.config import PipelineConfig


class CompatRunnerHelpersTest(unittest.TestCase):
    def test_build_output_paths_matches_naming_contract(self):
        cfg = SimpleNamespace(
            directory="/tmp/case_a",
            output_subdir="Coregistration",
        )

        output_dir, registered = build_output_paths(cfg)

        self.assertEqual(output_dir, Path("/tmp/case_a/Coregistration"))
        self.assertEqual(registered, Path("/tmp/case_a/Coregistration/registered_mask.tif"))

    def test_set_cfg_key_updates_nested_config_value(self):
        cfg = PipelineConfig()
        _set_cfg_key(cfg, "coarse_config.iters", 123)
        self.assertEqual(cfg.coarse_config.iters, 123)

    def test_set_cfg_key_raises_for_unknown_key(self):
        cfg = PipelineConfig()
        with self.assertRaises(AttributeError):
            _set_cfg_key(cfg, "coarse_config.unknown_field", 1)

    def test_resolve_eval_reference_returns_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            p = case_dir / "ICP_DNA.tiff"
            p.write_bytes(b"ok")

            resolved = resolve_eval_reference(case_dir, "ICP_DNA.tiff")
            self.assertEqual(resolved, p)

    def test_resolve_eval_reference_raises_for_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_eval_reference(case_dir, "missing.tiff")

    def test_pipeline_config_defaults_for_bspline_strategy(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.bspline_optimizer, "lbfgs2")
        self.assertEqual(cfg.bspline_metric, "mattes")
        self.assertTrue(cfg.bspline_deterministic_sampling)
        self.assertEqual(cfg.bspline_mesh_scale_factors, [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
