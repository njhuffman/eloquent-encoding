import React from "react";

export function EloEstimate(
  { estimate, bands, moves, minMoves }:
  { estimate: { posterior: number[]; meanElo: number; mapElo: number } | null;
    bands: number[]; moves: number; minMoves: number },
) {
  if (!estimate) {
    const need = Math.max(minMoves - moves, 1);
    return (
      <div className="card">
        <h3 className="card__title">Your estimated rating</h3>
        <p className="empty-hint">Play {need} more move{need === 1 ? "" : "s"} to estimate your rating.</p>
      </div>
    );
  }
  const max = Math.max(...estimate.posterior, 1e-9);
  return (
    <div className="card">
      <h3 className="card__title">Your estimated rating</h3>
      <div className="estimate__rating">≈ {estimate.mapElo}</div>
      <div className="estimate__from">from {moves} of your moves</div>
      <div className="estimate__chart">
        {bands.map((b, i) => (
          <div key={b} className={"estimate__bar" + (b === estimate.mapElo ? " is-map" : "")}
               style={{ height: `${(estimate.posterior[i] / max) * 70}px` }} />
        ))}
      </div>
      <div className="estimate__labels">{bands.map((b) => <span key={b}>{b}</span>)}</div>
      <p className="estimate__caveat">Indicative match, not a calibrated rating.</p>
    </div>
  );
}
