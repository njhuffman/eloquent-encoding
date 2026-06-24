import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine, error, books } = useEngine();
  const [elo, setElo] = useState(1500);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 16 }}>
      <h1>Eloquent Bot</h1>
      {error && <p style={{ color: "crimson" }}>Failed to load model: {error}</p>}
      {!engine && !error && <p>Loading model…</p>}
      <Controls elo={elo} setElo={setElo} temperature={temperature} setTemperature={setTemperature}
                playerColor={playerColor} setPlayerColor={setPlayerColor} />
      <BoardPanel engine={engine} elo={elo} temperature={temperature} books={books}
                  playerColor={playerColor} />
    </div>
  );
}
