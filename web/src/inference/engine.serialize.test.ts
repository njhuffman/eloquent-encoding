import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { mulberry32 } from "./rng";

// Faithfully mimics onnxruntime-web's wasm backend: a SINGLE global run guard
// shared across all sessions that throws "Session mismatch" if two run() calls
// overlap (see node_modules/onnxruntime-web ...jsep.mjs: `if (f.Yc) ... Session mismatch`).
// onnxruntime-node has no such guard, which is why the node parity tests never caught this.
let running = false;

function makeSession(outputName: string, dataLen: number) {
  return {
    async run(_feeds: Record<string, any>) {
      if (running) throw new Error("Session mismatch"); // overlap → reproduce the real failure
      running = true;
      try {
        await Promise.resolve(); // yield: opens the overlap window a racing run would hit
        await Promise.resolve();
        return { [outputName]: { data: new Float32Array(dataLen) } };
      } finally {
        running = false;
      }
    },
  };
}

let createCalls = 0;
const fakeOrt = {
  InferenceSession: {
    async create(_p: string) {
      const i = createCalls++; // load() creates encode, fromHead, toHead in order
      if (i % 3 === 0) return makeSession("squares", 64 * 256);
      if (i % 3 === 1) return makeSession("from_logits", 64);
      return makeSession("to_logits", 64);
    },
  },
  Tensor: class {
    constructor(public type: string, public data: any, public dims: number[]) {}
  },
};

describe("Engine ORT-run serialization", () => {
  it("serializes runs so concurrent chooseMove + topMoves(distributions) don't trip the single-run guard", async () => {
    const eng = await Engine.load(fakeOrt as any, { encode: "e", fromHead: "f", toHead: "t", valueHead: "v" }, { nEloBuckets: 40 });
    const board = new Chess();
    // Fire both concurrently — exactly what dropping a piece does (botMove + the panel effect).
    // Without serialization the second run starts while the first is mid-flight → "Session mismatch".
    const [mv, dist] = await Promise.all([
      eng.chooseMove(board, 1500, { temperature: 1, greedy: false, rand: mulberry32(1) }),
      eng.distributions(board, 1500).then(async (d) => {
        await d.toLogits(12);
        return d;
      }),
    ]);
    expect(mv.from).toBeTruthy();
    expect(mv.to).toBeTruthy();
    expect(dist.fromLogits.length).toBe(64);
  });
});
