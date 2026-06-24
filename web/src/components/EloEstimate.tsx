import React from "react";

export function EloEstimate(
  { estimate, bands, moves, minMoves }:
  { estimate: { posterior: number[]; meanElo: number; mapElo: number } | null;
    bands: number[]; moves: number; minMoves: number },
) {
  if (!estimate) {
    const need = Math.max(minMoves - moves, 1);
    return (
      <div style={{ minWidth: 210 }}>
        <h3 style={{ marginBottom: 6 }}>Your estimated rating</h3>
        <p style={{ color: "#999", margin: 0, fontSize: 13 }}>
          Play {need} more move{need === 1 ? "" : "s"} to estimate your rating.
        </p>
      </div>
    );
  }
  const max = Math.max(...estimate.posterior, 1e-9);
  return (
    <div style={{ minWidth: 210 }}>
      <h3 style={{ marginBottom: 2 }}>Your estimated rating</h3>
      <div style={{ fontSize: 22, fontWeight: 700 }}>≈ {Math.round(estimate.meanElo / 50) * 50}</div>
      <div style={{ color: "#777", fontSize: 12, marginBottom: 6 }}>from {moves} of your moves</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 72 }}>
        {bands.map((b, i) => (
          <div key={b} style={{
            flex: 1, height: `${(estimate.posterior[i] / max) * 70}px`, minHeight: 1,
            background: b === estimate.mapElo ? "#2e7d32" : "#4a90d9", borderRadius: "2px 2px 0 0",
          }} />
        ))}
      </div>
      <div style={{ display: "flex", gap: 3, fontSize: 9, color: "#777", marginTop: 2 }}>
        {bands.map((b) => <span key={b} style={{ flex: 1, textAlign: "center" }}>{b}</span>)}
      </div>
      <p style={{ color: "#999", fontSize: 11, marginTop: 4 }}>Indicative match, not a calibrated rating.</p>
    </div>
  );
}
