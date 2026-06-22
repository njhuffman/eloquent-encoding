import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import { Engine } from "./inference/engine";

export function useEngine() {
  const [engine, setEngine] = useState<Engine | null>(null);
  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(base + "model_meta.json").then((r) => r.json()).then((meta) =>
      Engine.load(ort as any, {
        encode: base + "encode_int8.onnx",
        fromHead: base + "from_head_int8.onnx",
        toHead: base + "to_head_int8.onnx",
      }, { nEloBuckets: meta.n_elo_buckets }).then(setEngine));
  }, []);
  return { engine };
}
