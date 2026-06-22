import type { Chess } from "chess.js";

const C = 18;
// plane index for a piece: white p,n,b,r,q,k -> 0..5 ; black -> 6..11
const PIECE_PLANE: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

export function squareToIndex(name: string): number {
  const file = name.charCodeAt(0) - 97;      // 'a' -> 0
  const rank = name.charCodeAt(1) - 49;      // '1' -> 0
  return rank * 8 + file;
}
export function indexToSquare(i: number): string {
  return String.fromCharCode(97 + (i % 8)) + String.fromCharCode(49 + Math.floor(i / 8));
}

export function boardToTensor(board: Chess): Float32Array {
  const t = new Float32Array(64 * C);
  // pieces: board.board() is rank 8..1, file a..h
  const rows = board.board();
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = rows[r][f];
      if (!piece) continue;
      const sq = (7 - r) * 8 + f;            // rows[0] is rank 8 -> rank index 7
      const plane = PIECE_PLANE[piece.type] + (piece.color === "w" ? 0 : 6);
      t[sq * C + plane] = 1.0;
    }
  }
  const white = board.turn() === "w";
  for (let s = 0; s < 64; s++) if (white) t[s * C + 12] = 1.0;     // plane 12 side-to-move
  // castling: chess.js getCastlingRights
  const wc = board.getCastlingRights("w"), bc = board.getCastlingRights("b");
  const setPlane = (plane: number, on: boolean) => { if (on) for (let s = 0; s < 64; s++) t[s * C + plane] = 1.0; };
  setPlane(13, wc.k); setPlane(14, wc.q); setPlane(15, bc.k); setPlane(16, bc.q);
  // en passant
  const fen = board.fen().split(" ");
  const ep = fen[3];
  if (ep && ep !== "-") t[squareToIndex(ep) * C + 17] = 1.0;
  return t;
}
