// Responsive board edge length: the container width (floored), capped at `max`, never below 1
// (so a 0/NaN measurement can't reach react-chessboard).
export function boardSizeFor(containerWidth: number, max = 480): number {
  return Math.max(1, Math.min(Math.floor(containerWidth), max));
}
