"""Unit tests for JEPA dashboard helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.dashboard.orchestration import is_safe_model_name, next_missing_stage
from jepa.training_loop import run_training_epochs


class TestTripletMeanNNegWithinMargin(unittest.TestCase):
    def test_mean_n_neg_within_margin_matches_active_count(self) -> None:
        import torch

        from jepa.architectures.chess_jepa_v1 import jepa_triplet_vicreg_loss

        B, K, D = 2, 3, 1
        z_online = torch.randn(B, D)
        z_hat = torch.zeros(B, D)
        z_pos = torch.zeros(B, D)
        z_negs = torch.zeros(B, K, D)
        z_negs[0, 1, 0] = 0.5
        z_negs[0, 2, 0] = 2.0
        z_negs[1, :, 0] = 3.0

        _, metrics = jepa_triplet_vicreg_loss(
            z_online,
            z_hat,
            z_pos,
            z_negs,
            margin_alpha=1.0,
            vicreg_var_coef=0.1,
            vicreg_std_target=1.0,
        )
        # d_pos=0; active if d_neg < 1: row0 has d in {0, 0.25, 4} -> 2 actives; row1 has 9 -> 0
        self.assertAlmostEqual(metrics["mean_n_neg_within_margin"], 1.0, places=5)


class TestOrchestration(unittest.TestCase):
    def test_safe_model_name(self) -> None:
        self.assertTrue(is_safe_model_name("example"))
        self.assertTrue(is_safe_model_name("bakeoff_og_1"))
        self.assertFalse(is_safe_model_name("../x"))
        self.assertFalse(is_safe_model_name(""))

    def test_next_missing_stage(self) -> None:
        tmp = Path(self._temp_ckpt_dir())
        name = "tmodel"
        spec = {"name": name, "stages": [{"x": 1}, {"x": 2}], "checkpoint_dir": str(tmp)}
        s, _ = next_missing_stage(spec)
        self.assertEqual(s, 0)
        stage_checkpoint_path(tmp, name, 0).touch()
        s, _ = next_missing_stage(spec)
        self.assertEqual(s, 1)
        stage_checkpoint_path(tmp, name, 1).touch()
        s, _ = next_missing_stage(spec)
        self.assertEqual(s, 2)
        stage_checkpoint_path(tmp, name, 2).touch()
        s, msg = next_missing_stage(spec)
        self.assertIsNone(s)
        self.assertIn("complete", msg.lower())

    def _temp_ckpt_dir(self) -> str:
        import tempfile

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class TestMetricsJsonl(unittest.TestCase):
    def test_run_training_writes_jsonl(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        tmp = Path(self._temp_ckpt_dir())
        metrics_path = tmp / "metrics" / "m_stage_1_epochs.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        class _M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def trainable_parameters(self):
                return self.parameters()

            def ema_update_target(self, _m: float) -> None:
                pass

            def train(self, mode: bool = True) -> None:
                pass

            def eval(self) -> None:
                pass

            def forward_online(self, board_t, elo):
                return board_t.mean(dim=(1, 2, 3)), elo * 0

            def forward_target(self, board):
                return board.mean(dim=(1, 2, 3))

            def forward_target_stack(self, boards_bk):
                b, k, h, w, c = boards_bk.shape
                flat = boards_bk.reshape(b * k, h, w, c)
                z = self.forward_target(flat)
                return z.reshape(b, k, -1)

        m = _M()
        H, W, C = 8, 8, 12
        n = 4
        board = torch.randn(n, H, W, C)
        pos = torch.randn(n, H, W, C)
        negs = torch.randn(n, 3, H, W, C)
        elo = torch.randn(n)
        ds = TensorDataset(board, pos, negs, elo)
        loader = DataLoader(ds, batch_size=2)

        def patched_loss(zo, zh, zp, zn, **kw):
            _ = (zo, zh, zp, zn)
            return (m.w ** 2).sum(), {
                "pct_active": 1.0,
                "mean_n_neg_within_margin": 0.5,
                "pct_pos_beats_hardest_neg": 2.0,
                "vicreg_std_mean": 0.5,
            }

        import jepa.training_loop as tl

        orig = tl.jepa_triplet_vicreg_loss
        tl.jepa_triplet_vicreg_loss = patched_loss
        try:
            run_training_epochs(
                m,
                train_loader=loader,
                val_loader=loader,
                device=torch.device("cpu"),
                epochs=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                use_amp=False,
                ema_momentum=0.9,
                margin_alpha=0.2,
                vicreg_var_coef=0.1,
                vicreg_std_target=1.0,
                log_interval=0,
                metrics_jsonl_path=metrics_path,
                metrics_run_meta={"model": "m", "stage": 1},
            )
        finally:
            tl.jepa_triplet_vicreg_loss = orig

        lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        row = json.loads(lines[0])
        self.assertEqual(row["epoch"], 1)
        self.assertIn("train_loss", row)
        self.assertIn("val_loss", row)
        self.assertIn("train_mean_n_neg_within_margin", row)
        self.assertIn("val_mean_n_neg_within_margin", row)

    def _temp_ckpt_dir(self) -> str:
        import tempfile

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


if __name__ == "__main__":
    unittest.main()
