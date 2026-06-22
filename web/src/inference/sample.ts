export function maskedSoftmax(
  logits: Float32Array | number[], legal: boolean[], temperature: number,
): Float32Array {
  const n = logits.length;
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (legal[i] && logits[i] / t > max) max = logits[i] / t;
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (!legal[i]) continue;
    const e = Math.exp(logits[i] / t - max);
    out[i] = e; sum += e;
  }
  if (sum > 0) for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function pickIndex(probs: Float32Array, opts: { greedy: boolean; rand?: () => number }): number {
  if (opts.greedy) {
    let best = 0, bestv = -Infinity;
    for (let i = 0; i < probs.length; i++) if (probs[i] > bestv) { bestv = probs[i]; best = i; }
    return best;
  }
  const r = (opts.rand ?? Math.random)();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) return i; }
  return probs.length - 1;
}
