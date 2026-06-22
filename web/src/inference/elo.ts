export function eloToBucket(elo: number, n: number): number {
  if (elo > 0) return Math.min(Math.max(Math.floor(elo / 100), 0), n - 1);
  return n; // null / unknown-elo index
}
