import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine, error } = useEngine();
  const [elo, setElo] = useState(1500);
  const [temperature, setTemperature] = useState(1.0);
  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <h1>Eloquent Bot</h1>
      {error && <p style={{color:"crimson"}}>Failed to load model: {error}</p>}
      {!engine && !error && <p>Loading model…</p>}
      <Controls elo={elo} setElo={setElo} temperature={temperature} setTemperature={setTemperature} />
      <BoardPanel engine={engine} elo={elo} temperature={temperature} />
    </div>
  );
}
