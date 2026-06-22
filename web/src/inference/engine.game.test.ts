import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { mulberry32 } from "./rng";
import fixtures from "./__fixtures__/cases.json";

describe("engine self-play", () => {
  it("plays only legal moves and terminates", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx", fromHead: "public/from_head_int8.onnx", toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const board = new Chess();
    const rand = mulberry32(7);
    let plies = 0;
    while (!board.isGameOver() && plies < 400) {
      const mv = await eng.chooseMove(board, 1500, { temperature: 1.0, greedy: false, rand });
      const res = board.move(mv);
      expect(res).not.toBeNull();   // legal
      plies++;
    }
    expect(plies).toBeGreaterThan(2);
  }, 60000);
});
