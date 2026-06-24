import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { indexToSquare } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";
import { squareToIndex } from "./boardTensor";
import { maskedSoftmax } from "./sample";
import { legalFromMask, legalToMask } from "./legal";

describe("Engine (int8) parity vs python fixtures", () => {
  it("greedy top move matches python top_move_uci", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
      valueHead: "public/value_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    for (const c of fixtures.cases) {
      const board = new Chess(c.fen);
      const mv = await eng.chooseMove(board, c.elo, { temperature: 1, greedy: true });
      // python top_move_uci is from+to (+promotion); compare squares
      expect(mv.from + mv.to).toBe(c.top_move_uci.slice(0, 4));
    }
  });
});

function softmax3(a: number[] | Float32Array) {
  const m = Math.max(a[0], a[1], a[2]);
  const e = [Math.exp(a[0] - m), Math.exp(a[1] - m), Math.exp(a[2] - m)];
  const s = e[0] + e[1] + e[2];
  return [e[0] / s, e[1] / s, e[2] / s];
}

describe("Engine.moveProbsByElo", () => {
  it("equals the masked from/to product from distributions, and is a probability", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
      valueHead: "public/value_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const board = new Chess(); // start position
    const elo = 1500;
    const { fromLogits, toLogits } = await eng.distributions(board, elo);
    const fromIdx = squareToIndex("e2"), toIdx = squareToIndex("e4");
    const pFrom = maskedSoftmax(fromLogits, legalFromMask(board), 1.0)[fromIdx];
    const tl = await toLogits(fromIdx);
    const pTo = maskedSoftmax(tl, legalToMask(board, fromIdx), 1.0)[toIdx];
    const expected = pFrom * pTo;
    const [got] = await eng.moveProbsByElo(board, { from: "e2", to: "e4" }, [elo]);
    expect(got).toBeCloseTo(expected, 6);
    expect(got).toBeGreaterThan(0);
    expect(got).toBeLessThanOrEqual(1);
  });
});

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
      // int8 quantization introduces ~1% error vs float32 PyTorch fixtures; use precision 1 (±0.05)
      expect(v.loss).toBeCloseTo(ref[0], 1);
      expect(v.draw).toBeCloseTo(ref[1], 1);
      expect(v.win).toBeCloseTo(ref[2], 1);
    }
  });
});
