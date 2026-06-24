import React, { useEffect, useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";
import { useMediaQuery } from "./useMediaQuery";

export function App() {
  const { engine, error, books } = useEngine();
  const [botElo, setBotElo] = useState(1500);
  const [analysisElo, setAnalysisElo] = useState(1500);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [gameStarted, setGameStarted] = useState(false); // locks the bot-elo control once a move is played

  // Settings toolbar collapses on mobile; open by default on desktop.
  const isMobile = useMediaQuery("(max-width: 700px)");
  const [settingsOpen, setSettingsOpen] = useState(!isMobile);
  useEffect(() => setSettingsOpen(!isMobile), [isMobile]);

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Eloquent Bot</h1>
        <div className="app__subtitle">A human-like chess bot — see how each rating would move, and estimate your own.</div>
      </header>
      {error && <p className="app__error">Failed to load model: {error}</p>}
      {!engine && !error && <p className="app__status">Loading model…</p>}
      <details className="toolbar-wrap" open={settingsOpen}
               onToggle={(e) => setSettingsOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>Settings ▾</summary>
        <Controls
          botElo={botElo} setBotElo={setBotElo} botEloLocked={gameStarted}
          analysisElo={analysisElo} setAnalysisElo={setAnalysisElo}
          showAnalysis={showAnalysis} setShowAnalysis={setShowAnalysis}
          temperature={temperature} setTemperature={setTemperature}
          playerColor={playerColor} setPlayerColor={setPlayerColor}
        />
      </details>
      <BoardPanel
        engine={engine} botElo={botElo} analysisElo={analysisElo} showAnalysis={showAnalysis}
        temperature={temperature} books={books} playerColor={playerColor}
        onGameStartedChange={setGameStarted}
      />
    </div>
  );
}
