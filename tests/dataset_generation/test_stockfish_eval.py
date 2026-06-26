import numpy as np
from chess.engine import Cp, Mate
from dataset_generation.stockfish_eval import (
    CP_CLAMP, STATIC_NA, clamp_cp, parse_static_eval, score_to_cp_mate,
    select_rows, open_or_create_sidecar, pending_positions, write_records,
)

def test_clamp_cp():
    assert clamp_cp(120) == 120
    assert clamp_cp(50000) == CP_CLAMP
    assert clamp_cp(-50000) == -CP_CLAMP

def test_parse_static_eval_normal():
    assert parse_static_eval("NNUE evaluation        +0.49 (white side)") == 49
    assert parse_static_eval("NNUE evaluation        -1.23 (white side)") == -123
    assert parse_static_eval("Final evaluation: +0.00 (white side)") == 0

def test_parse_static_eval_in_check():
    assert parse_static_eval("Final evaluation: none (in check)") is None

def test_score_to_cp_mate():
    assert score_to_cp_mate(Cp(120)) == (120, 0)
    assert score_to_cp_mate(Cp(50000)) == (CP_CLAMP, 0)   # clamped
    assert score_to_cp_mate(Mate(3)) == (CP_CLAMP, 3)
    assert score_to_cp_mate(Mate(-2)) == (-CP_CLAMP, -2)


def _attrs():
    return {"source_h5": "x.h5", "source_n_rows": 100, "depth": 8, "sample_n": -1,
            "seed": 0, "perspective": "STM", "wdl_order": "loss,draw,win",
            "cp_clamp": CP_CLAMP, "stockfish_version": "test"}

def test_select_rows():
    assert np.array_equal(select_rows(5, None, 0), np.arange(5))
    s = select_rows(100, 10, 0)
    assert len(s) == 10 and len(set(s.tolist())) == 10 and list(s) == sorted(s)
    assert np.array_equal(select_rows(100, 10, 0), select_rows(100, 10, 0))  # seeded

def test_sidecar_create_pending_write_resume(tmp_path):
    p = str(tmp_path / "sc.h5")
    rows = np.arange(6, dtype=np.int64)
    f = open_or_create_sidecar(p, rows, _attrs())
    assert list(pending_positions(f)) == [0, 1, 2, 3, 4, 5]
    write_records(f, [0, 2], [
        {"cp": 30, "mate": 0, "static_cp": 25, "wdl": (100, 300, 600)},
        {"cp": -32000, "mate": -1, "static_cp": STATIC_NA, "wdl": (700, 200, 100)},
    ])
    assert list(pending_positions(f)) == [1, 3, 4, 5]
    assert f["sf_cp"][0] == 30 and f["sf_mate"][2] == -1
    assert list(f["sf_wdl"][0]) == [100, 300, 600]
    f.close()
    # reload: resumes (done preserved), attrs validated
    f2 = open_or_create_sidecar(p, rows, _attrs())
    assert list(pending_positions(f2)) == [1, 3, 4, 5]
    f2.close()

def test_sidecar_rejects_mismatch(tmp_path):
    p = str(tmp_path / "sc.h5")
    open_or_create_sidecar(p, np.arange(6, dtype=np.int64), _attrs()).close()
    try:
        open_or_create_sidecar(p, np.arange(7, dtype=np.int64), _attrs())  # different rows
        assert False, "expected mismatch error"
    except ValueError:
        pass

def test_sidecar_rejects_attr_mismatch(tmp_path):
    import numpy as np
    p = str(tmp_path / "sc.h5")
    rows = np.arange(6, dtype=np.int64)
    open_or_create_sidecar(p, rows, _attrs()).close()
    bad = _attrs(); bad["depth"] = 12  # same rows, different depth
    try:
        open_or_create_sidecar(p, rows, bad)
        assert False, "expected attr mismatch error"
    except ValueError:
        pass
