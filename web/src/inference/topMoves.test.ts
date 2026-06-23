import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { topMoves } from "./topMoves";
import fixtures from "./__fixtures__/cases.json";

describe("topMoves", () => {
  it("returns sorted legal moves with probabilities summing <= 1", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx", fromHead: "public/from_head_int8.onnx", toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const moves = await topMoves(eng, new Chess(), 1500, 5);
    expect(moves.length).toBe(5);
    for (let i = 1; i < moves.length; i++) expect(moves[i - 1].prob).toBeGreaterThanOrEqual(moves[i].prob);
    expect(moves.every((m) => new Chess().move(m.san) !== null)).toBe(true);
  }, 60000);
});
