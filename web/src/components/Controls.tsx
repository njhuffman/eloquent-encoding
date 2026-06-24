import React from "react";

export function Controls({
  botElo, setBotElo, botEloLocked, analysisElo, setAnalysisElo,
  showAnalysis, setShowAnalysis, temperature, setTemperature, playerColor, setPlayerColor,
}: {
  botElo: number; setBotElo: (n: number) => void; botEloLocked: boolean;
  analysisElo: number; setAnalysisElo: (n: number) => void;
  showAnalysis: boolean; setShowAnalysis: (b: boolean) => void;
  temperature: number; setTemperature: (n: number) => void;
  playerColor: "w" | "b"; setPlayerColor: (c: "w" | "b") => void;
}) {
  return (
    <div className="toolbar">
      <label className="control">
        <span className="control__label">Bot elo: {botElo}{botEloLocked ? " 🔒" : ""}</span>
        <input type="range" min={1000} max={1900} step={100} value={botElo} disabled={botEloLocked}
               onChange={(e) => setBotElo(Number(e.target.value))} />
      </label>
      <label className={"control" + (showAnalysis ? "" : " control--dim")}>
        <span className="control__label">Analysis elo: {analysisElo}</span>
        <input type="range" min={1000} max={1900} step={100} value={analysisElo} disabled={!showAnalysis}
               onChange={(e) => setAnalysisElo(Number(e.target.value))} />
      </label>
      <label className="check">
        <input type="checkbox" checked={showAnalysis} onChange={(e) => setShowAnalysis(e.target.checked)} />
        Show analysis
      </label>
      <label className="control">
        <span className="control__label">Temperature: {temperature.toFixed(1)}</span>
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
      <div className="control">
        <span className="control__label">Play as</span>
        <div className="seg">
          <button className={"seg__opt" + (playerColor === "w" ? " is-active" : "")}
                  onClick={() => setPlayerColor("w")}>White</button>
          <button className={"seg__opt" + (playerColor === "b" ? " is-active" : "")}
                  onClick={() => setPlayerColor("b")}>Black</button>
        </div>
      </div>
    </div>
  );
}
