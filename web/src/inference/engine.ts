import type { Chess } from "chess.js";
import { boardToTensor, indexToSquare } from "./boardTensor";
import { eloToBucket } from "./elo";
import { legalFromMask, legalToMask } from "./legal";
import { maskedSoftmax, pickIndex } from "./sample";

type Session = { run(feeds: Record<string, any>): Promise<Record<string, { data: Float32Array }>> };
type OrtLike = {
  InferenceSession: { create(p: string | ArrayBuffer): Promise<Session> };
  Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => any;
};

// onnxruntime-web's wasm backend has a SINGLE global run guard: if a second
// run() starts while another is in flight (e.g. the bot's chooseMove racing the
// panel's topMoves), it throws "Session mismatch". The guard is module-global in
// the runtime — shared across every session and every Engine instance — so this
// serialization queue is module-level too, funnelling all runs strictly one at a
// time. (onnxruntime-node has no such guard, which is why the parity tests missed it.)
let runQueue: Promise<unknown> = Promise.resolve();

function serializedRun(
  session: Session, feeds: Record<string, any>,
): Promise<Record<string, { data: Float32Array }>> {
  const result = runQueue.then(() => session.run(feeds));
  runQueue = result.then(() => undefined, () => undefined); // keep the chain alive on success or failure
  return result;
}

export class Engine {
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

  // Serialize a single ORT run behind any in-flight run (module-global queue).
  private run(session: Session, feeds: Record<string, any>): Promise<Record<string, { data: Float32Array }>> {
    return serializedRun(session, feeds);
  }

  private elo(elo: number) {
    return new this.ort.Tensor("int64", BigInt64Array.from([BigInt(eloToBucket(elo, this.nEloBuckets))]), [1]);
  }

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

  async policy(board: Chess, elo: number) {
    const { squares: sq } = await this.encode(board);
    const eloT = this.elo(elo);
    const fl = (await this.run(this.fh, { squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromProbs = maskedSoftmax(fl, legalFromMask(board), 1.0);
    const fromSq = pickIndex(fromProbs, { greedy: true });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.run(this.th, { squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toProbs = maskedSoftmax(tl, legalToMask(board, fromSq), 1.0);
    const toSq = pickIndex(toProbs, { greedy: true });
    return { fromProbs, fromSq, toProbs, toSq };
  }

  async distributions(board: Chess, elo: number) {
    const { squares: sq } = await this.encode(board);
    const eloT = this.elo(elo);
    const fromLogits = (await this.run(this.fh, { squares: sq, elo_idx: eloT }))["from_logits"].data;
    const toLogits = async (fromSq: number) => {
      const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
      return (await this.run(this.th, { squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    };
    return { fromLogits, toLogits };
  }

  async chooseMove(board: Chess, elo: number, opts: { temperature: number; greedy?: boolean; rand?: () => number }) {
    const { squares: sq } = await this.encode(board);
    const eloT = this.elo(elo);
    const fl = (await this.run(this.fh, { squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromSq = pickIndex(maskedSoftmax(fl, legalFromMask(board), opts.temperature),
                             { greedy: !!opts.greedy, rand: opts.rand });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.run(this.th, { squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toSq = pickIndex(maskedSoftmax(tl, legalToMask(board, fromSq), opts.temperature),
                           { greedy: !!opts.greedy, rand: opts.rand });
    const from = indexToSquare(fromSq), to = indexToSquare(toSq);
    // promotion: if a pawn reaches the last rank, default to queen (matches play.py)
    const needsPromo = board.moves({ verbose: true })
      .some((m) => m.from === from && m.to === to && m.promotion);
    return needsPromo ? { from, to, promotion: "q" as const } : { from, to };
  }
}
