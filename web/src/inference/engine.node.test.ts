import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { indexToSquare } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("Engine (int8) parity vs python fixtures", () => {
  it("greedy top move matches python top_move_uci", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    for (const c of fixtures.cases) {
      const board = new Chess(c.fen);
      const mv = await eng.chooseMove(board, c.elo, { temperature: 1, greedy: true });
      // python top_move_uci is from+to (+promotion); compare squares
      expect(mv.from + mv.to).toBe(c.top_move_uci.slice(0, 4));
    }
  });
});
