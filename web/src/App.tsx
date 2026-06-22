import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";

export function App() {
  const { engine } = useEngine();
  const [elo] = useState(1500);
  const [temperature] = useState(1.0);
  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <h1>Eloquent Bot</h1>
      {!engine && <p>Loading model…</p>}
      <BoardPanel engine={engine} elo={elo} temperature={temperature} />
    </div>
  );
}
