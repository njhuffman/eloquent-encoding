export type Color = "w" | "b";

export const botColorOf = (c: Color): Color => (c === "w" ? "b" : "w");

export const boardOrientationOf = (c: Color): "white" | "black" => (c === "w" ? "white" : "black");

// The bot (White) makes the opening move only when the human chose Black and no moves have been played.
export const botShouldOpen = (c: Color, historyLength: number): boolean => c === "b" && historyLength === 0;
