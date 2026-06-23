import React from "react";

export function ThinkingPanel({ moves }: { moves: { san: string; prob: number }[] }) {
  const max = moves.length ? moves[0].prob : 1;
  return (
    <div style={{ minWidth: 180 }}>
      <h3>Model's top moves</h3>
      {moves.map((m) => (
        <div key={m.san} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 48 }}>{m.san}</span>
          <div style={{ background: "#4a90d9", height: 12, width: `${(m.prob / max) * 100}%` }} />
          <span>{(m.prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
