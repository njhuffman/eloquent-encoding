import React from "react";

type Move = { uci?: string; san: string; prob: number };

export function ThinkingPanel({ title, moves, highlightUci, emptyHint }: {
  title: string;
  moves: Move[];
  highlightUci?: string; // mark the move that was actually played (the bot's choice)
  emptyHint?: string;
}) {
  const max = moves.length ? moves[0].prob : 1;
  return (
    <div style={{ minWidth: 210 }}>
      <h3 style={{ marginBottom: 6 }}>{title}</h3>
      {moves.length === 0 && <p style={{ color: "#999", margin: 0 }}>{emptyHint ?? "—"}</p>}
      {moves.map((m) => {
        const chosen = !!m.uci && m.uci === highlightUci;
        return (
          <div key={m.uci ?? m.san} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 56, fontWeight: chosen ? 700 : 400 }}>
              {chosen ? "▶ " : ""}{m.san}
            </span>
            <div style={{ background: chosen ? "#2e7d32" : "#4a90d9", height: 12, width: `${(m.prob / max) * 100}%` }} />
            <span>{(m.prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
