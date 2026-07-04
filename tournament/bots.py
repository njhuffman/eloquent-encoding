"""Bot registry + adapter factory. Loads bots.yaml (stable ids + type + params + optional ref_elo)
and builds a Player for any entry. Shared/expensive resources (the Maia2 net) are cached on the
factory so many maia2 bots don't reload it."""
from __future__ import annotations
from pathlib import Path
import yaml

_LC0 = "/tmp/lc0/build/release/lc0"


def load_registry(path: str | Path) -> list[dict]:
    entries = yaml.safe_load(Path(path).read_text()) or []
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate bot ids in {path}")
    return entries


class BotFactory:
    """Builds Player instances from registry entries; caches the Maia2 model."""

    def __init__(self, device: str = "cuda", lc0: str = _LC0):
        self.device = device
        self.lc0 = lc0
        self._maia2 = None  # (model, prep), lazily loaded
        self._mb: dict = {}  # checkpoint -> (model, arch), shared across multiband bots

    def _maia2_model(self):
        if self._maia2 is None:
            from style_policy.maia2_bot import load_maia2
            self._maia2 = load_maia2("rapid", device=("gpu" if str(self.device).startswith("cuda") else "cpu"))
        return self._maia2

    def build(self, entry: dict):
        t = entry["type"]; p = dict(entry.get("params", {})); seed = int(p.pop("seed", 0))
        if t == "multiband":
            from style_policy.multiband_bot import MultiBandBot
            ckpt = p["checkpoint"]
            if ckpt not in self._mb:
                import torch
                from style_policy.multiband_policy import MultiBandPolicy
                c = torch.load(ckpt, map_location=self.device)
                m = MultiBandPolicy.from_config(c["architecture"]).to(self.device); m.load_state_dict(c["model"]); m.eval()
                self._mb[ckpt] = (m, c["architecture"])
            model, arch = self._mb[ckpt]
            return MultiBandBot(ckpt, int(p["band"]), device=self.device,
                                temperature=float(p.get("temperature", 1.0)), seed=seed, model=model, arch=arch)
        if t == "maia1":
            from style_policy.maia1_bot import Maia1Bot
            return Maia1Bot(p["weights"], lc0=self.lc0, temperature=float(p.get("temperature", 1.0)),
                            nodes=int(p.get("nodes", 1)), seed=seed)
        if t == "maia2":
            from style_policy.maia2_bot import Maia2Bot
            model, prep = self._maia2_model()
            return Maia2Bot(model, prep, self_elo=int(p["elo"]), seed=seed)
        if t == "stockfish":
            from tournament.stockfish_bot import StockfishBot
            return StockfishBot(skill_level=p.get("skill_level"), elo=p.get("elo"),
                                depth=int(p.get("depth", 8)), movetime=p.get("movetime"), seed=seed)
        if t == "expectimax":
            from style_policy.search_bot import ExpectimaxBot
            return ExpectimaxBot(p["checkpoint"], int(p["elo"]), int(p["depth"]),
                                 width=int(p.get("width", 4)), device=self.device, seed=seed)
        raise ValueError(f"unknown bot type {t!r}")
