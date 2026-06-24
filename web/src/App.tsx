import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine, error, books } = useEngine();
  const [botElo, setBotElo] = useState(1500);
  const [analysisElo, setAnalysisElo] = useState(1500);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [gameStarted, setGameStarted] = useState(false); // locks the bot-elo control once a move is played
  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 16 }}>
      <h1>Eloquent Bot</h1>
      {error && <p style={{ color: "crimson" }}>Failed to load model: {error}</p>}
      {!engine && !error && <p>Loading model…</p>}
      <Controls
        botElo={botElo} setBotElo={setBotElo} botEloLocked={gameStarted}
        analysisElo={analysisElo} setAnalysisElo={setAnalysisElo}
        showAnalysis={showAnalysis} setShowAnalysis={setShowAnalysis}
        temperature={temperature} setTemperature={setTemperature}
        playerColor={playerColor} setPlayerColor={setPlayerColor}
      />
      <BoardPanel
        engine={engine} botElo={botElo} analysisElo={analysisElo} showAnalysis={showAnalysis}
        temperature={temperature} books={books} playerColor={playerColor}
        onGameStartedChange={setGameStarted}
      />
    </div>
  );
}
