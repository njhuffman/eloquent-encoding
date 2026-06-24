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
    <div className="wdlbar" style={{ height }} title="White / draw / black win probability">
      {order.map((s, i) => (
        <div key={i} className={`wdlbar__seg is-${s.kind}`}
             style={{ flexGrow: Math.max(s.prob, 0.0001), flexBasis: 0 }}>
          {wdl && s.prob >= 0.08 ? Math.round(s.prob * 100) : ""}
        </div>
      ))}
    </div>
  );
}
