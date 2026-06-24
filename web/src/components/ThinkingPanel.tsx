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
    <div className="card">
      <h3 className="card__title">{title}</h3>
      {moves.length === 0 && <p className="empty-hint">{emptyHint ?? "—"}</p>}
      <div className="movelist">
        {moves.map((m) => {
          const chosen = !!m.uci && m.uci === highlightUci;
          return (
            <div key={m.uci ?? m.san} className="move">
              <span className={"move__san" + (chosen ? " is-chosen" : "")}>{chosen ? "▶ " : ""}{m.san}</span>
              <div className={"move__bar" + (chosen ? " is-chosen" : "")} style={{ width: `${(m.prob / max) * 100}%` }} />
              <span className="move__pct">{(m.prob * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
