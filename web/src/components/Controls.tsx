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
    <div style={{ display: "flex", gap: 24, margin: "12px 0", alignItems: "center", flexWrap: "wrap" }}>
      <label>Bot elo: {botElo}{botEloLocked ? " 🔒" : ""}
        <input type="range" min={600} max={2400} step={100} value={botElo} disabled={botEloLocked}
               onChange={(e) => setBotElo(Number(e.target.value))} />
      </label>
      <label style={{ opacity: showAnalysis ? 1 : 0.5 }}>Analysis elo: {analysisElo}
        <input type="range" min={600} max={2400} step={100} value={analysisElo} disabled={!showAnalysis}
               onChange={(e) => setAnalysisElo(Number(e.target.value))} />
      </label>
      <label>
        <input type="checkbox" checked={showAnalysis} onChange={(e) => setShowAnalysis(e.target.checked)} />
        {" "}Show analysis
      </label>
      <label>Temperature: {temperature.toFixed(1)}
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
      <span>
        Play as:{" "}
        <button onClick={() => setPlayerColor("w")} disabled={playerColor === "w"}>White</button>{" "}
        <button onClick={() => setPlayerColor("b")} disabled={playerColor === "b"}>Black</button>
      </span>
    </div>
  );
}
