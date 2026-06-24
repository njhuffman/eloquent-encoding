# Web WDL bar + player color — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vertical WDL bar (P(white)/P(draw)/P(black)) left of the board and a White/Black player-color picker, switching the deployed web bot from `base_64M` to the value-headed `wdl_16M` model.

**Architecture:** Export a value-head ONNX graph from `wdl_16M` (the encode graph gains a `cls` output; a tiny `value_head` graph maps `cls + elo → 3 WDL logits`). The web `Engine` gains `value()`. `BoardPanel` is generalized off its hardcoded-White assumptions to a `playerColor` prop; a new `WDLBar` component renders the bar with the player's color at the bottom.

**Tech Stack:** Python (PyTorch, onnxruntime, onnxruntime.quantization), TypeScript/React (react-chessboard, chess.js), onnxruntime-web/-node, vitest.

## Global Constraints

- The value head exists **only** in `style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt` (`d_model=256`, `head_hidden=512`, `elo_dim=32`, `n_elo_buckets=40`). The deployed bot switches to this checkpoint.
- WDL logits are ordered **(loss=0, draw=1, win=2)** from the **side-to-move's** perspective.
- The encoder returns `(cls, squares)` in that order (`policy.encoder(bt)` → `(cls, squares)`).
- ONNX output names: encode → `["squares", "cls"]`; from_head → `from_logits`; to_head → `to_logits`; value_head → `value_logits`.
- All web ORT runs go through the existing module-global `serializedRun` queue (onnxruntime-web single-run guard).
- Python scripts run as modules: `python -m scripts.export_onnx`, never `python scripts/export_onnx.py`.
- Commit only the four `web/public/*_int8.onnx`, `model_meta.json`, and `cases.json` regenerated from `wdl_16M`. No Git LFS.
- Git footer on every commit:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`.

---

## File Structure

- `style_policy/onnx_export.py` — MODIFY: `EncodeExport` emits `(squares, cls)`; add `ValueHeadExport`; `build_export_modules` returns it.
- `scripts/export_onnx.py` — MODIFY: export + quantize 4 graphs; `value_head_int8.onnx` in `model_meta.json`.
- `scripts/gen_web_fixtures.py` — MODIFY: add `value_logits` to each case.
- `tests/style_policy/test_onnx_export.py` — MODIFY: fix tuple unpack; add value-head parity.
- `web/public/{encode,from_head,to_head,value_head}_int8.onnx`, `web/public/model_meta.json`, `web/src/inference/__fixtures__/cases.json` — REGENERATE from `wdl_16M`.
- `web/src/inference/engine.ts` — MODIFY: load `value_head`; `encode()` returns `{squares, cls}`; add `value()`.
- `web/src/useEngine.ts` — MODIFY: pass `valueHead` URL.
- `web/src/inference/engine.node.test.ts` — MODIFY: add value-parity test.
- `web/src/components/WDLBar.tsx` — CREATE: `arrangeWDL` helper + `WDLBar` component.
- `web/src/components/WDLBar.test.ts` — CREATE: unit test for `arrangeWDL`.
- `web/src/playerColor.ts` (+ `playerColor.test.ts`) — CREATE: pure color helpers (`botColorOf`, `boardOrientationOf`, `botShouldOpen`).
- `web/src/App.tsx` — MODIFY: `playerColor` state.
- `web/src/components/Controls.tsx` — MODIFY: White/Black picker.
- `web/src/components/BoardPanel.tsx` — MODIFY: generalize for `playerColor`; integrate `WDLBar` + value computation.

---

## Task 1: ONNX export — `cls` output + value-head graph

**Files:**
- Modify: `style_policy/onnx_export.py`
- Modify: `scripts/export_onnx.py`
- Test: `tests/style_policy/test_onnx_export.py`

**Interfaces:**
- Consumes: `policy.encoder(bt) -> (cls, squares)`, `policy.value_head(cls, elo_idx=...) -> (B,3)`.
- Produces: `EncodeExport.forward(board_tensor) -> (squares, cls)`; `ValueHeadExport.forward(cls, elo_idx) -> value_logits`; `build_export_modules(policy) -> (enc, fh, th, vh)`; `export_fp32`/`quantize_and_check` handle 4 graphs; `value_head.onnx` input names `["cls","elo_idx"]`, output `["value_logits"]`.

- [ ] **Step 1: Update existing wrapper-parity test to expect the tuple + value head**

In `tests/style_policy/test_onnx_export.py`, replace `test_export_wrappers_match_eager` with:

```python
def test_export_wrappers_match_eager():
    policy = BasePolicy.from_config(CFG).eval()
    enc, fh, th, vh = build_export_modules(policy)
    bt = _board_tensor()
    with torch.no_grad():
        cls_ref, squares_ref = policy.encoder(bt)
        squares, cls = enc(bt)
        assert torch.allclose(squares, squares_ref, atol=1e-5)
        assert torch.allclose(cls, cls_ref, atol=1e-5)
        elo = torch.tensor([12, 18], dtype=torch.long)
        assert torch.allclose(fh(squares, elo), policy.from_head(squares, elo_idx=elo), atol=1e-5)
        fsq = torch.tensor([0, 4], dtype=torch.long)
        assert torch.allclose(th(squares, fsq, elo), policy.to_head(squares, fsq, elo_idx=elo), atol=1e-5)
        assert torch.allclose(vh(cls, elo), policy.value_head(cls, elo_idx=elo), atol=1e-5)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/style_policy/test_onnx_export.py::test_export_wrappers_match_eager -x -q`
Expected: FAIL — `build_export_modules` returns 3 values (cannot unpack into 4) / `enc(bt)` returns a single tensor.

- [ ] **Step 3: Modify `style_policy/onnx_export.py`**

Replace the file body (keep the module docstring) with:

```python
class EncodeExport(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, board_tensor: torch.Tensor):
        cls, squares = self.encoder(board_tensor)
        return squares, cls


class FromHeadExport(nn.Module):
    def __init__(self, from_head: nn.Module):
        super().__init__()
        self.from_head = from_head

    def forward(self, squares: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares, elo_idx=elo_idx)


class ToHeadExport(nn.Module):
    def __init__(self, to_head: nn.Module):
        super().__init__()
        self.to_head = to_head

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq, elo_idx=elo_idx)


class ValueHeadExport(nn.Module):
    def __init__(self, value_head: nn.Module):
        super().__init__()
        self.value_head = value_head

    def forward(self, cls: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.value_head(cls, elo_idx=elo_idx)


def build_export_modules(policy):
    policy.eval()
    # nn.TransformerEncoder's nested-tensor fast path uses data-dependent ops that don't trace.
    enc = policy.encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "enable_nested_tensor"):
        enc.encoder.enable_nested_tensor = False
    return (EncodeExport(policy.encoder).eval(),
            FromHeadExport(policy.from_head).eval(),
            ToHeadExport(policy.to_head).eval(),
            ValueHeadExport(policy.value_head).eval())
```

- [ ] **Step 4: Modify `scripts/export_onnx.py` — `export_fp32`**

In `export_fp32`, change the unpack and add the value-head export. Replace from the `enc, fh, th = build_export_modules(policy)` line through the three `torch.onnx.export(...)` calls with:

```python
    enc, fh, th, vh = build_export_modules(policy)
    d = int(ck["architecture"]["d_model"])

    bt = torch.from_numpy(board_tensor_for_fen(chess.STARTING_FEN))
    with torch.no_grad():
        squares, cls = enc(bt)
    elo = torch.tensor([15], dtype=torch.long)
    fsq = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(enc, (bt,), str(out_dir / "encode.onnx"), opset_version=OPSET,
                      input_names=["board_tensor"], output_names=["squares", "cls"])
    torch.onnx.export(fh, (squares, elo), str(out_dir / "from_head.onnx"), opset_version=OPSET,
                      input_names=["squares", "elo_idx"], output_names=["from_logits"])
    torch.onnx.export(th, (squares, fsq, elo), str(out_dir / "to_head.onnx"), opset_version=OPSET,
                      input_names=["squares", "from_sq", "elo_idx"], output_names=["to_logits"])
    torch.onnx.export(vh, (cls, elo), str(out_dir / "value_head.onnx"), opset_version=OPSET,
                      input_names=["cls", "elo_idx"], output_names=["value_logits"])
```

- [ ] **Step 5: Modify `scripts/export_onnx.py` — `quantize_and_check` + `main` meta**

In `quantize_and_check`, change the loop tuple to include `value_head`:

```python
    for name in ("encode", "from_head", "to_head", "value_head"):
```

In `main`, change the `model_meta.json` `files` list to:

```python
         "files": ["encode_int8.onnx", "from_head_int8.onnx", "to_head_int8.onnx", "value_head_int8.onnx"]}))
```

- [ ] **Step 6: Add an fp32 value-head ONNX parity test**

Append to `tests/style_policy/test_onnx_export.py`:

```python
def test_fp32_value_head_parity(tmp_path):
    ck = torch.load(CKPT, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"]); policy.load_state_dict(ck["model"], strict=False); policy.eval()
    export_fp32(CKPT, tmp_path)
    enc = ort.InferenceSession(str(tmp_path / "encode.onnx"))
    vh = ort.InferenceSession(str(tmp_path / "value_head.onnx"))
    for fen in FENS:
        bt = board_tensor_for_fen(fen); elo = np.array([15], dtype=np.int64)
        with torch.no_grad():
            cls_ref, _ = policy.encoder(torch.from_numpy(bt))
            vlog_ref = policy.value_head(cls_ref, elo_idx=torch.from_numpy(elo)).numpy()
        outs = {o.name: i for i, o in enumerate(enc.get_outputs())}
        cls = enc.run(None, {"board_tensor": bt})[outs["cls"]]
        vlog = vh.run(None, {"cls": cls, "elo_idx": elo})[0]
        assert np.allclose(vlog, vlog_ref, atol=1e-4)
```

(`CKPT`/`FENS`/`ort`/`np` already exist in the file. `base_64M` lacks trained value-head weights, but parity compares the ONNX graph to the same eager module — random init is fine.)

- [ ] **Step 7: Run the export tests**

Run: `python -m pytest tests/style_policy/test_onnx_export.py -x -q`
Expected: PASS (all tests, including the existing fp32/int8/fixtures tests which read output index `[0]` = `squares`, still valid).

- [ ] **Step 8: Commit**

```bash
git add style_policy/onnx_export.py scripts/export_onnx.py tests/style_policy/test_onnx_export.py
git commit -m "feat(onnx): export cls output + value_head graph"
```

---

## Task 2: Regenerate deployed artifacts + fixtures from `wdl_16M`

**Files:**
- Modify: `scripts/gen_web_fixtures.py`
- Test: `tests/style_policy/test_onnx_export.py` (extend `test_fixtures_written`)
- Regenerate (commit): `web/public/{encode,from_head,to_head,value_head}_int8.onnx`, `web/public/model_meta.json`, `web/src/inference/__fixtures__/cases.json`

**Interfaces:**
- Consumes: Task 1's `export_onnx` (4 graphs); `policy.encoder`/`policy.value_head`.
- Produces: `cases.json` cases each with a `"value_logits"` length-3 list (loss/draw/win); deployed int8 graphs + meta from `wdl_16M`.

- [ ] **Step 1: Extend the fixtures test to require `value_logits`**

In `tests/style_policy/test_onnx_export.py`, change `test_fixtures_written` to also assert the value field, and point it at `wdl_16M` so it exercises a trained value head:

```python
WDL_CKPT = "style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt"

def test_fixtures_written(tmp_path, monkeypatch):
    from scripts.gen_web_fixtures import build_cases
    cases = build_cases(WDL_CKPT, FENS, elo=1500)
    assert len(cases["cases"]) == len(FENS)
    c = cases["cases"][0]
    assert len(c["board_tensor"]) == 8 * 8 * 18
    assert len(c["from_logits"]) == 64 and len(c["legal_from"]) == 64
    assert len(c["value_logits"]) == 3
    assert c["bucket"] == 15
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/style_policy/test_onnx_export.py::test_fixtures_written -x -q`
Expected: FAIL — `KeyError: 'value_logits'` (build_cases doesn't emit it yet).

- [ ] **Step 3: Modify `scripts/gen_web_fixtures.py` — emit `value_logits`**

In `build_cases`, inside the `with torch.no_grad():` block, capture `cls` and compute value. Replace the encode line and add a value computation; the current `_, sq = policy.encoder(bt)` becomes:

```python
            cls, sq = policy.encoder(bt)
            fl = policy.from_head(sq, elo_idx=torch.tensor([bucket]))[0]
            from_sq = int(fl.masked_fill(~torch.tensor(_bits(legal_from_u64(board))), float("-inf")).argmax())
            tl = policy.to_head(sq, torch.tensor([from_sq]), elo_idx=torch.tensor([bucket]))[0]
            to_legal = _bits(legal_to_u64(board, from_sq))
            to_sq = int(tl.masked_fill(~torch.tensor(to_legal), float("-inf")).argmax())
            value_logits = policy.value_head(cls, elo_idx=torch.tensor([bucket]))[0]
```

And add `"value_logits": value_logits.tolist(),` to the `cases.append({...})` dict (e.g. after `"to_logits"`).

- [ ] **Step 4: Run the fixtures test**

Run: `python -m pytest tests/style_policy/test_onnx_export.py::test_fixtures_written -x -q`
Expected: PASS.

- [ ] **Step 5: Regenerate the deployed int8 artifacts from `wdl_16M`**

Run: `python -m scripts.export_onnx --checkpoint style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt`
Expected output ends with `int8 deploy artifacts -> web/public {'size_bytes': ...}`.
Verify: `ls web/public/*.onnx` shows `encode_int8.onnx from_head_int8.onnx to_head_int8.onnx value_head_int8.onnx`, and `python -c "import json;print(json.load(open('web/public/model_meta.json'))['files'])"` lists all four.

- [ ] **Step 6: Regenerate `cases.json` from `wdl_16M`**

Run: `python -m scripts.gen_web_fixtures --checkpoint style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt`
Expected: `wrote web/src/inference/__fixtures__/cases.json`.
Verify: `python -c "import json;c=json.load(open('web/src/inference/__fixtures__/cases.json'))['cases'][0];print(len(c['value_logits']), c['top_move_uci'])"` prints `3 <uci>`.

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_web_fixtures.py tests/style_policy/test_onnx_export.py \
        web/public/encode_int8.onnx web/public/from_head_int8.onnx \
        web/public/to_head_int8.onnx web/public/value_head_int8.onnx \
        web/public/model_meta.json web/src/inference/__fixtures__/cases.json
git commit -m "feat(web): switch deployed model to wdl_16M + value fixtures"
```

---

## Task 3: `Engine.value` + load value-head + useEngine wiring

**Files:**
- Modify: `web/src/inference/engine.ts`
- Modify: `web/src/useEngine.ts`
- Test: `web/src/inference/engine.node.test.ts`

**Interfaces:**
- Consumes: `value_head_int8.onnx` (`cls,elo_idx → value_logits`), encode graph output `cls`, `cases.json` cases with `value_logits`.
- Produces: `Engine.load(ort, { encode, fromHead, toHead, valueHead }, { nEloBuckets })`; `Engine.value(board, elo) -> Promise<{loss,draw,win}>` (softmax, side-to-move perspective, order loss/draw/win).

- [ ] **Step 1: Write the failing value-parity test**

Append to `web/src/inference/engine.node.test.ts` (inside the existing `describe`, or a new `describe`):

```ts
function softmax3(a: number[] | Float32Array) {
  const m = Math.max(a[0], a[1], a[2]);
  const e = [Math.exp(a[0] - m), Math.exp(a[1] - m), Math.exp(a[2] - m)];
  const s = e[0] + e[1] + e[2];
  return [e[0] / s, e[1] / s, e[2] / s];
}

describe("Engine.value parity vs python fixtures", () => {
  it("WDL softmax matches python value_logits", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
      valueHead: "public/value_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    for (const c of fixtures.cases as any[]) {
      const v = await eng.value(new Chess(c.fen), c.elo);
      const ref = softmax3(c.value_logits);
      expect(v.loss).toBeCloseTo(ref[0], 2);
      expect(v.draw).toBeCloseTo(ref[1], 2);
      expect(v.win).toBeCloseTo(ref[2], 2);
    }
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd web && npx vitest run src/inference/engine.node.test.ts`
Expected: FAIL — `Engine.load` rejects the `valueHead` URL / `eng.value` is not a function.

- [ ] **Step 3: Modify `web/src/inference/engine.ts` — constructor + load**

Change the constructor signature to add the value session, and `load` to create 4 sessions:

```ts
  private constructor(
    private ort: OrtLike, private enc: Session, private fh: Session, private th: Session,
    private vh: Session, private nEloBuckets: number,
  ) {}

  static async load(ort: OrtLike,
                    urls: { encode: string; fromHead: string; toHead: string; valueHead: string },
                    meta: { nEloBuckets: number }): Promise<Engine> {
    const [enc, fh, th, vh] = await Promise.all([
      ort.InferenceSession.create(urls.encode),
      ort.InferenceSession.create(urls.fromHead),
      ort.InferenceSession.create(urls.toHead),
      ort.InferenceSession.create(urls.valueHead),
    ]);
    return new Engine(ort, enc, fh, th, vh, meta.nEloBuckets);
  }
```

- [ ] **Step 4: Modify `web/src/inference/engine.ts` — `encode` returns `{squares, cls}` + `value`**

Replace the `private async squares(board)` method with an `encode` method returning both tensors, and add `value`:

```ts
  private async encode(board: Chess): Promise<{ squares: any; cls: any }> {
    const bt = boardToTensor(board);
    const out = await this.run(this.enc, { board_tensor: new this.ort.Tensor("float32", bt, [1, 8, 8, 18]) });
    return { squares: out["squares"], cls: out["cls"] };
  }

  async value(board: Chess, elo: number): Promise<{ loss: number; draw: number; win: number }> {
    const { cls } = await this.encode(board);
    const l = (await this.run(this.vh, { cls, elo_idx: this.elo(elo) }))["value_logits"].data;
    const m = Math.max(l[0], l[1], l[2]);
    const e = [Math.exp(l[0] - m), Math.exp(l[1] - m), Math.exp(l[2] - m)];
    const s = e[0] + e[1] + e[2];
    return { loss: e[0] / s, draw: e[1] / s, win: e[2] / s };
  }
```

Then update the three callers that did `const sq = await this.squares(board)` — in `policy`, `distributions`, and `chooseMove`, replace that line with `const { squares: sq } = await this.encode(board);` (everything else in those methods is unchanged).

- [ ] **Step 5: Run the engine tests**

Run: `cd web && npx vitest run src/inference/engine.node.test.ts`
Expected: PASS — both the existing greedy-top-move parity and the new value parity.

- [ ] **Step 6: Wire `useEngine.ts` to pass the value-head URL**

In `web/src/useEngine.ts`, add the `valueHead` URL to the `Engine.load` call:

```ts
      .then((meta) => Engine.load(ort as any, {
        encode: base + "encode_int8.onnx",
        fromHead: base + "from_head_int8.onnx",
        toHead: base + "to_head_int8.onnx",
        valueHead: base + "value_head_int8.onnx",
      }, { nEloBuckets: meta.n_elo_buckets }))
```

- [ ] **Step 7: Run the full web suite to confirm nothing else broke**

Run: `cd web && npx vitest run`
Expected: PASS (all existing tests; the policy path still reads `out["squares"]`).

- [ ] **Step 8: Commit**

```bash
git add web/src/inference/engine.ts web/src/useEngine.ts web/src/inference/engine.node.test.ts
git commit -m "feat(web): Engine.value + load value-head graph"
```

---

## Task 4: `WDLBar` component + arrangement helper

**Files:**
- Create: `web/src/components/WDLBar.tsx`
- Test: `web/src/components/WDLBar.test.ts`

**Interfaces:**
- Consumes: a WDL triple `{loss,draw,win}` (side-to-move perspective), `sideToMove: 'w'|'b'`, `playerColor: 'w'|'b'`.
- Produces: `arrangeWDL(wdl, sideToMove, playerColor) -> { top, mid, bottom }` where each is `{ kind: 'white'|'black'|'draw'; prob: number }`, ordered top→bottom with the player's color at the bottom and draw in the middle; `WDLBar` React component.

- [ ] **Step 1: Write the failing test**

Create `web/src/components/WDLBar.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { arrangeWDL } from "./WDLBar";

describe("arrangeWDL", () => {
  const wdl = { loss: 0.1, draw: 0.3, win: 0.6 };

  it("white player, white to move: white-win at bottom", () => {
    const a = arrangeWDL(wdl, "w", "w");
    expect(a.bottom).toEqual({ kind: "white", prob: 0.6 });
    expect(a.top).toEqual({ kind: "black", prob: 0.1 });
    expect(a.mid).toEqual({ kind: "draw", prob: 0.3 });
  });

  it("black player, white to move: black-win at bottom", () => {
    const a = arrangeWDL(wdl, "w", "b");
    expect(a.bottom).toEqual({ kind: "black", prob: 0.1 });
    expect(a.top).toEqual({ kind: "white", prob: 0.6 });
  });

  it("applies the side-to-move flip (black to move)", () => {
    // black to move: wdl.win is BLACK's win prob
    const a = arrangeWDL(wdl, "b", "w");
    expect(a.bottom).toEqual({ kind: "white", prob: 0.1 }); // white-win = loss for black mover
    expect(a.top).toEqual({ kind: "black", prob: 0.6 });
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd web && npx vitest run src/components/WDLBar.test.ts`
Expected: FAIL — cannot resolve `./WDLBar` / `arrangeWDL` not exported.

- [ ] **Step 3: Create `web/src/components/WDLBar.tsx`**

```tsx
import React from "react";

export type WDL = { loss: number; draw: number; win: number };
type Seg = { kind: "white" | "black" | "draw"; prob: number };

// Convert a side-to-move WDL into three segments ordered top->bottom, with the
// player's own color at the BOTTOM so the bar matches the flipped board.
export function arrangeWDL(
  wdl: WDL, sideToMove: "w" | "b", playerColor: "w" | "b",
): { top: Seg; mid: Seg; bottom: Seg } {
  const pWhite = sideToMove === "w" ? wdl.win : wdl.loss;
  const pBlack = sideToMove === "w" ? wdl.loss : wdl.win;
  const playerWhite = playerColor === "w";
  const bottom: Seg = playerWhite ? { kind: "white", prob: pWhite } : { kind: "black", prob: pBlack };
  const top: Seg = playerWhite ? { kind: "black", prob: pBlack } : { kind: "white", prob: pWhite };
  return { top, mid: { kind: "draw", prob: wdl.draw }, bottom };
}

const COLORS: Record<Seg["kind"], string> = { white: "#f0f0f0", black: "#333", draw: "#9e9e9e" };
const LABELC: Record<Seg["kind"], string> = { white: "#222", black: "#eee", draw: "#fff" };

export function WDLBar(
  { wdl, sideToMove, playerColor, height = 480 }:
  { wdl: WDL | null; sideToMove: "w" | "b"; playerColor: "w" | "b"; height?: number },
) {
  const a = wdl
    ? arrangeWDL(wdl, sideToMove, playerColor)
    : { top: { kind: "black", prob: 0 }, mid: { kind: "draw", prob: 1 }, bottom: { kind: "white", prob: 0 } } as
        { top: Seg; mid: Seg; bottom: Seg };
  const order: Seg[] = [a.top, a.mid, a.bottom];
  return (
    <div style={{ display: "flex", flexDirection: "column", width: 28, height,
                  border: "1px solid #ccc", borderRadius: 4, overflow: "hidden" }}
         title="White / draw / black win probability">
      {order.map((s, i) => (
        <div key={i} style={{ flexGrow: Math.max(s.prob, 0.0001), flexBasis: 0,
                              background: COLORS[s.kind], display: "flex",
                              alignItems: "center", justifyContent: "center",
                              fontSize: 10, color: LABELC[s.kind] }}>
          {wdl && s.prob >= 0.08 ? Math.round(s.prob * 100) : ""}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Run the test**

Run: `cd web && npx vitest run src/components/WDLBar.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/WDLBar.tsx web/src/components/WDLBar.test.ts
git commit -m "feat(web): WDLBar component + arrangeWDL helper"
```

---

## Task 5: Player color — pure helpers, App state, Controls picker, BoardPanel generalization

**Files:**
- Create: `web/src/playerColor.ts`
- Test: `web/src/playerColor.test.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/Controls.tsx`
- Modify: `web/src/components/BoardPanel.tsx`

**Interfaces:**
- Consumes: `Engine`, `OpeningBookSet` (unchanged).
- Produces: pure helpers `botColorOf(c)`, `boardOrientationOf(c)`, `botShouldOpen(c, historyLength)`; `BoardPanel` accepts `playerColor: 'w' | 'b'`; human moves only on `playerColor` turns; bot plays the opposite color; board orientation + new-game-on-color-change + bot-opens-when-Black.

**Note on testing:** the web vitest config is node-only (`environment: "node"`, `include: ["src/**/*.test.ts"]`); there is no DOM/testing-library and `.test.tsx` files aren't collected. So the player-color *logic* is extracted into a pure module and unit-tested (matching the repo's `undo.ts`/`arrangeWDL` convention); the React wiring is verified by typecheck + build (Step 7), as `BoardPanel` has no component test today.

- [ ] **Step 1: Write the failing pure-helper test**

Create `web/src/playerColor.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { botColorOf, boardOrientationOf, botShouldOpen } from "./playerColor";

describe("player color helpers", () => {
  it("botColorOf is the opposite color", () => {
    expect(botColorOf("w")).toBe("b");
    expect(botColorOf("b")).toBe("w");
  });
  it("boardOrientationOf maps to react-chessboard strings", () => {
    expect(boardOrientationOf("w")).toBe("white");
    expect(boardOrientationOf("b")).toBe("black");
  });
  it("bot opens only when the human is Black and the board is fresh", () => {
    expect(botShouldOpen("b", 0)).toBe(true);
    expect(botShouldOpen("b", 1)).toBe(false);
    expect(botShouldOpen("w", 0)).toBe(false);
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd web && npx vitest run src/playerColor.test.ts`
Expected: FAIL — cannot resolve `./playerColor`.

- [ ] **Step 2b: Create `web/src/playerColor.ts`**

```ts
export type Color = "w" | "b";

export const botColorOf = (c: Color): Color => (c === "w" ? "b" : "w");

export const boardOrientationOf = (c: Color): "white" | "black" => (c === "w" ? "white" : "black");

// The bot (White) makes the opening move only when the human chose Black and no moves have been played.
export const botShouldOpen = (c: Color, historyLength: number): boolean => c === "b" && historyLength === 0;
```

Run: `cd web && npx vitest run src/playerColor.test.ts`
Expected: PASS.

- [ ] **Step 3: Add `playerColor` state to `App.tsx`**

```tsx
import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine, error, books } = useEngine();
  const [elo, setElo] = useState(1500);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 16 }}>
      <h1>Eloquent Bot</h1>
      {error && <p style={{ color: "crimson" }}>Failed to load model: {error}</p>}
      {!engine && !error && <p>Loading model…</p>}
      <Controls elo={elo} setElo={setElo} temperature={temperature} setTemperature={setTemperature}
                playerColor={playerColor} setPlayerColor={setPlayerColor} />
      <BoardPanel engine={engine} elo={elo} temperature={temperature} books={books}
                  playerColor={playerColor} />
    </div>
  );
}
```

- [ ] **Step 4: Add the White/Black picker to `Controls.tsx`**

```tsx
import React from "react";

export function Controls({ elo, setElo, temperature, setTemperature, playerColor, setPlayerColor }: {
  elo: number; setElo: (n: number) => void; temperature: number; setTemperature: (n: number) => void;
  playerColor: "w" | "b"; setPlayerColor: (c: "w" | "b") => void;
}) {
  return (
    <div style={{ display: "flex", gap: 24, margin: "12px 0", alignItems: "center", flexWrap: "wrap" }}>
      <label>Elo: {elo}
        <input type="range" min={600} max={2400} step={100} value={elo}
               onChange={(e) => setElo(Number(e.target.value))} />
      </label>
      <label>Temperature: {temperature.toFixed(1)}
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
      <span>
        Play as:{" "}
        <button onClick={() => setPlayerColor("w")} disabled={playerColor === "w"}>White</button>{" "}
        <button onClick={() => setPlayerColor("b")} disabled={playerColor === "b"}>Black</button>
      </span>
    </div>
  );
}
```

- [ ] **Step 5: Generalize `BoardPanel.tsx` for `playerColor`**

Make these edits to `web/src/components/BoardPanel.tsx`:

(a) Signature — add `playerColor` and import the helpers (`import { botColorOf, boardOrientationOf, botShouldOpen } from "../playerColor";` at the top):

```tsx
export function BoardPanel({ engine, elo, temperature, books, playerColor }:
  { engine: Engine | null; elo: number; temperature: number; books: OpeningBookSet | null; playerColor: "w" | "b" }) {
```

(b) Near the top of the component body add the bot color (used by the analysis effect):

```tsx
  const botColor = botColorOf(playerColor);
```

(c) In the analysis `useEffect`, replace the White-specific gates with `playerColor`/`botColor` and add `playerColor` to the dependency array:

```tsx
      // Your move (only meaningful when it's the human's turn and the game is live)
      if (!g.isGameOver() && g.turn() === playerColor) {
        const ym = await topMoves(engine, new Chess(g.fen()), elo, 5);
        if (!cancelled) setYourMoves(ym);
      } else if (!cancelled) {
        setYourMoves([]);
      }
      // Bot's last move: reconstruct the position just before the bot's last move
      const verbose = g.history({ verbose: true });
      let lastBotIdx = -1;
      for (let i = verbose.length - 1; i >= 0; i--) { if (verbose[i].color === botColor) { lastBotIdx = i; break; } }
      if (lastBotIdx >= 0) {
        const pre = new Chess();
        for (let i = 0; i < lastBotIdx; i++) pre.move(verbose[i].san);
        const list = await topMoves(engine, pre, elo, 5);
        const mv = verbose[lastBotIdx];
        if (!cancelled) setBotAnalysis({ list, chosenUci: mv.from + mv.to });
      } else if (!cancelled) {
        setBotAnalysis(null);
      }
```

Change the effect deps from `[engine, fen, elo]` to `[engine, fen, elo, playerColor, botColor]`.

(d) In `onDrop`, reject drags when it isn't the human's turn — add at the top of the callback (after the `thinking` guard):

```tsx
    if (gameRef.current.turn() !== playerColor) return false;
```

(e) After `botMove` is defined, add a ref so color/new-game effects can call the latest without retriggering on elo/temperature changes:

```tsx
  const botMoveRef = useRef(botMove);
  botMoveRef.current = botMove;
```

(f) Replace `newGame` so it opens for Black, and add the two color effects. Replace the existing `newGame` callback with:

```tsx
  const newGame = useCallback(() => {
    if (thinking) return;
    gameRef.current = new Chess();
    setLastMove(null);
    setFen(gameRef.current.fen());
    if (playerColor === "b") void botMoveRef.current();
  }, [thinking, playerColor]);
```

And add, after the analysis effect:

```tsx
  // Picking a color starts a fresh game.
  useEffect(() => {
    gameRef.current = new Chess();
    setLastMove(null);
    setFen(gameRef.current.fen());
  }, [playerColor]);

  // If the human is Black, the bot (White) opens once the board is fresh + engine is ready.
  useEffect(() => {
    if (engine && botShouldOpen(playerColor, gameRef.current.history().length) &&
        !gameRef.current.isGameOver()) {
      void botMoveRef.current();
    }
  }, [engine, playerColor]);
```

(g) Add `boardOrientation` to `<Chessboard>`:

```tsx
          boardOrientation={boardOrientationOf(playerColor)}
```

- [ ] **Step 6: Run the full web suite + typecheck**

Run: `cd web && npx vitest run && npx tsc --noEmit`
Expected: PASS, no type errors (the player-color helpers are unit-tested; the React wiring is verified by typecheck + the Task 6 build).

- [ ] **Step 7: Commit**

```bash
git add web/src/playerColor.ts web/src/playerColor.test.ts web/src/App.tsx web/src/components/Controls.tsx web/src/components/BoardPanel.tsx
git commit -m "feat(web): play as White or Black (board flips, bot opens for Black)"
```

---

## Task 6: Integrate `WDLBar` into `BoardPanel`

**Files:**
- Modify: `web/src/components/BoardPanel.tsx`

**Interfaces:**
- Consumes: `Engine.value` (Task 3), `WDLBar`/`WDL` (Task 4), `playerColor` (Task 5).
- Produces: a live WDL bar left of the board, recomputed on every position change.

- [ ] **Step 1: Import WDLBar + add wdl state**

At the top of `BoardPanel.tsx` add:

```tsx
import { WDLBar, type WDL } from "./WDLBar";
```

With the other `useState` hooks add:

```tsx
  const [wdl, setWdl] = useState<WDL | null>(null);
```

- [ ] **Step 2: Compute the value in the analysis effect**

Inside the analysis `useEffect`'s async block (the one keyed on `[engine, fen, elo, playerColor, botColor]`), after the yourMoves/botAnalysis logic, add a value computation:

```tsx
      // WDL bar (side-to-move perspective; conditioned on the elo slider)
      try {
        const v = await engine.value(new Chess(g.fen()), elo);
        if (!cancelled) setWdl(v);
      } catch { if (!cancelled) setWdl(null); }
```

- [ ] **Step 3: Render the bar to the left of the board**

Change the outer board-row `<div>` so the WDL bar is the first child. Replace the opening of the returned JSX (the `<div style={{ display:"flex", gap:16, ...}}>` and its first child) so the structure is:

```tsx
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <WDLBar wdl={wdl} sideToMove={view.turn()} playerColor={playerColor} height={480} />
      <div style={{ width: 480 }}>
        <Chessboard
          position={fen}
          onPieceDrop={onDrop}
          arePiecesDraggable={!thinking}
          customSquareStyles={customSquareStyles}
          boardWidth={480}
          boardOrientation={playerColor === "w" ? "white" : "black"}
        />
        {/* …existing controls + status… */}
      </div>
      {/* …existing thinking panels column… */}
    </div>
```

(`view` is the already-present `const view = gameRef.current;`. Leave the controls row, game-over line, and thinking-panel column exactly as they are.)

- [ ] **Step 4: Run the full web suite + typecheck**

Run: `cd web && npx vitest run && npx tsc --noEmit`
Expected: PASS, no type errors.

- [ ] **Step 5: Build to confirm the bundle compiles**

Run: `cd web && npm run build`
Expected: build succeeds (Vite emits `dist/`).

- [ ] **Step 6: Commit**

```bash
git add web/src/components/BoardPanel.tsx
git commit -m "feat(web): render live WDL bar left of the board"
```

---

## Self-review notes

- **Spec coverage:** export `cls`+value graph (T1), switch deployed model + value fixtures (T2), `Engine.value` (T3), `WDLBar` w/ player-at-bottom + STM flip (T4), player color + board flip + bot-opens-Black (T5), bar integration + live recompute (T6). Value-parity, WDLBar-arrange, and BoardPanel-color tests all present.
- **Naming consistency:** ONNX names (`squares`,`cls`,`from_logits`,`to_logits`,`value_logits`), `Engine.value`, `arrangeWDL`, `playerColor`/`botColor` used identically across tasks.
- **Known follow-ups (out of scope):** value conditioned on the elo slider (proxy, not true side-to-move rating); the bot is now `wdl_16M` (weaker policy, accepted).
