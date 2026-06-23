import type { Chess } from "chess.js";
import { boardToTensor, indexToSquare, squareToIndex } from "./boardTensor";
import { eloToBucket } from "./elo";
import { legalFromMask, legalToMask } from "./legal";
import { maskedSoftmax, pickIndex } from "./sample";

type Session = { run(feeds: Record<string, any>): Promise<Record<string, { data: Float32Array }>> };
type OrtLike = {
  InferenceSession: { create(p: string | ArrayBuffer): Promise<Session> };
  Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => any;
};

export class Engine {
  private constructor(
    private ort: OrtLike, private enc: Session, private fh: Session, private th: Session,
    private nEloBuckets: number,
  ) {}

  static async load(ort: OrtLike, urls: { encode: string; fromHead: string; toHead: string },
                    meta: { nEloBuckets: number }): Promise<Engine> {
    const [enc, fh, th] = await Promise.all([
      ort.InferenceSession.create(urls.encode),
      ort.InferenceSession.create(urls.fromHead),
      ort.InferenceSession.create(urls.toHead),
    ]);
    return new Engine(ort, enc, fh, th, meta.nEloBuckets);
  }

  private elo(elo: number) {
    return new this.ort.Tensor("int64", BigInt64Array.from([BigInt(eloToBucket(elo, this.nEloBuckets))]), [1]);
  }

  private async squares(board: Chess) {
    const bt = boardToTensor(board);
    const out = await this.enc.run({ board_tensor: new this.ort.Tensor("float32", bt, [1, 8, 8, 18]) });
    return out["squares"];
  }

  async policy(board: Chess, elo: number) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fl = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromProbs = maskedSoftmax(fl, legalFromMask(board), 1.0);
    const fromSq = pickIndex(fromProbs, { greedy: true });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toProbs = maskedSoftmax(tl, legalToMask(board, fromSq), 1.0);
    const toSq = pickIndex(toProbs, { greedy: true });
    return { fromProbs, fromSq, toProbs, toSq };
  }

  async distributions(board: Chess, elo: number) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fromLogits = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const toLogits = async (fromSq: number) => {
      const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
      return (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    };
    return { fromLogits, toLogits };
  }

  async chooseMove(board: Chess, elo: number, opts: { temperature: number; greedy?: boolean; rand?: () => number }) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fl = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromSq = pickIndex(maskedSoftmax(fl, legalFromMask(board), opts.temperature),
                             { greedy: !!opts.greedy, rand: opts.rand });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toSq = pickIndex(maskedSoftmax(tl, legalToMask(board, fromSq), opts.temperature),
                           { greedy: !!opts.greedy, rand: opts.rand });
    const from = indexToSquare(fromSq), to = indexToSquare(toSq);
    // promotion: if a pawn reaches the last rank, default to queen (matches play.py)
    const needsPromo = board.moves({ verbose: true })
      .some((m) => m.from === from && m.to === to && m.promotion);
    return needsPromo ? { from, to, promotion: "q" as const } : { from, to };
  }
}
