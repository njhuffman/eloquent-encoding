// Responsive board edge length: the container width (floored), capped at `max`, never below 1
// (so a 0/NaN measurement can't reach react-chessboard).
export function boardSizeFor(containerWidth: number, max = 480): number {
  return Math.max(1, Math.min(Math.floor(containerWidth), max));
}

// The largest board that fits BOTH the container width and the viewport height (the board is
// square, so its height = its width). `reserve` leaves vertical room for the header/toolbar/nav,
// so the whole board stays on-screen — important on short/landscape phones.
export function fitBoardSize(hostWidth: number, viewportHeight: number, reserve = 160, max = 480): number {
  return boardSizeFor(Math.min(hostWidth, viewportHeight - reserve), max);
}
