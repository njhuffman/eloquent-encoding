import React from "react";

export function Controls({ elo, setElo, temperature, setTemperature }: {
  elo: number; setElo: (n: number) => void; temperature: number; setTemperature: (n: number) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 24, margin: "12px 0" }}>
      <label>Elo: {elo}
        <input type="range" min={600} max={2400} step={100} value={elo}
               onChange={(e) => setElo(Number(e.target.value))} />
      </label>
      <label>Temperature: {temperature.toFixed(1)}
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
    </div>
  );
}
