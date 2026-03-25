from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

BucketBy = Literal["white", "black", "both"]


@dataclass(frozen=True)
class StratumSpec:
    elo_min: int
    elo_max: int
    take_games: int
    samples_per_game: int
    stratum_seed: int


@dataclass(frozen=True)
class SourcePlan:
    """One month (or dump) to stream once, with its own Elo strata and quotas."""

    source: str
    strata: tuple[StratumSpec, ...]


@dataclass(frozen=True)
class Recipe:
    """`name` is the dataset basename; HDF5 is written as ``{output_dir}/{name}.h5``."""

    name: str
    master_seed: int
    time_control: str | None
    bucket_by: BucketBy
    skip_opening_plies: int
    exclude_single_legal_move: bool
    source_plans: tuple[SourcePlan, ...]

    def upper_bound_sample_rows(self) -> int:
        """
        Maximum HDF5 rows if every accepted game yields `samples_per_game` rows.
        Actual row count may be lower when a game has fewer candidate plies than
        `samples_per_game` (still counts as one accepted game toward take_games).
        """
        return sum(
            st.take_games * st.samples_per_game
            for plan in self.source_plans
            for st in plan.strata
        )

    def output_h5_path(self, output_dir: Path) -> Path:
        """Resolved path ``output_dir / f\"{name}.h5\"`` (directory need not exist yet)."""
        return (output_dir.expanduser().resolve() / f"{self.name}.h5")

    @staticmethod
    def _parse_name(raw: Any) -> str:
        if not isinstance(raw, str):
            raise TypeError("name must be a string")
        n = raw.strip()
        if not n:
            raise ValueError("name must be non-empty")
        if "/" in n or "\\" in n:
            raise ValueError("name must be a basename only, not a path (use --output-dir)")
        if n in (".", ".."):
            raise ValueError(f"invalid name: {n!r}")
        if n.endswith(".h5"):
            raise ValueError("name must not include a .h5 suffix; output is always {name}.h5")
        return n

    @staticmethod
    def _parse_strata_list(items: list[Any], *, where: str) -> tuple[StratumSpec, ...]:
        if not items:
            raise ValueError(f"{where}: strata must be non-empty")
        strata: list[StratumSpec] = []
        for i, s in enumerate(items):
            if not isinstance(s, dict):
                raise TypeError(f"{where}[{i}] must be a mapping")
            for k in ("elo_min", "elo_max", "take_games", "samples_per_game", "stratum_seed"):
                if k not in s:
                    raise KeyError(f"{where}[{i}] missing {k!r}")
            strata.append(
                StratumSpec(
                    elo_min=int(s["elo_min"]),
                    elo_max=int(s["elo_max"]),
                    take_games=int(s["take_games"]),
                    samples_per_game=int(s["samples_per_game"]),
                    stratum_seed=int(s["stratum_seed"]),
                )
            )
        return tuple(strata)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Recipe:
        required = [
            "name",
            "master_seed",
            "bucket_by",
            "skip_opening_plies",
            "exclude_single_legal_move",
        ]
        for k in required:
            if k not in d:
                raise KeyError(f"recipe missing required key: {k!r}")

        tc = d.get("time_control", None)
        if tc is not None and not isinstance(tc, str):
            raise TypeError("time_control must be a string or null/omitted")

        bb = d["bucket_by"]
        if bb not in ("white", "black", "both"):
            raise ValueError(f"bucket_by must be white|black|both, got {bb!r}")

        raw_plans = d.get("source_plans")
        if raw_plans is None:
            raise KeyError(
                "recipe missing required key: 'source_plans' "
                "(list of {source: <month key>, strata: [...]})"
            )
        if not isinstance(raw_plans, list) or not raw_plans:
            raise ValueError("source_plans must be a non-empty list")

        plans: list[SourcePlan] = []
        for bi, block in enumerate(raw_plans):
            if not isinstance(block, dict):
                raise TypeError(f"source_plans[{bi}] must be a mapping")
            if "source" not in block:
                raise KeyError(f"source_plans[{bi}] missing 'source'")
            if "strata" not in block:
                raise KeyError(f"source_plans[{bi}] missing 'strata'")
            src = str(block["source"]).strip()
            if not src:
                raise ValueError(f"source_plans[{bi}].source must be non-empty")
            st_list = block["strata"]
            if not isinstance(st_list, list):
                raise TypeError(f"source_plans[{bi}].strata must be a list")
            strata = Recipe._parse_strata_list(st_list, where=f"source_plans[{bi}].strata")
            plans.append(SourcePlan(source=src, strata=strata))

        return Recipe(
            name=Recipe._parse_name(d["name"]),
            master_seed=int(d["master_seed"]),
            time_control=tc,
            bucket_by=bb,  # type: ignore[arg-type]
            skip_opening_plies=int(d["skip_opening_plies"]),
            exclude_single_legal_move=bool(d["exclude_single_legal_move"]),
            source_plans=tuple(plans),
        )

    @staticmethod
    def load(path: Path) -> Recipe:
        """
        Load a YAML recipe (recommended; supports # comments and multi-line notes).
        Plain JSON files are valid YAML and still work.
        """
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            raise ValueError(f"recipe file is empty or non-document: {path}")
        if not isinstance(data, dict):
            raise TypeError("recipe root must be a mapping (YAML object / JSON object)")
        return Recipe.from_dict(data)
