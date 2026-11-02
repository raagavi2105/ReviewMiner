type Extractor = (t: string[], o: object) => Promise<{ data: Float32Array; dims: number[] }>;
let extractor: Promise<Extractor> | null = null;
let onProg: ((p: number) => void) | undefined;
export const loadEmbedder = (onProgress?: (p: number) => void) => {
  if (onProgress) onProg = onProgress;
  return (extractor ??= import('@xenova/transformers').then(({ pipeline, env }) => {
    env.allowLocalModels = false;
    return pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      progress_callback: (d: { status: string; progress?: number }) => { if (d.status === 'progress' && d.progress !== undefined) onProg?.(d.progress); },
    }) as unknown as Promise<Extractor>;
  }));
};

export async function embed(texts: string[], onProgress: (done: number) => void): Promise<number[][]> {
  const ex = await loadEmbedder(); const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 32) {
    const batch = texts.slice(i, i + 32);
    const t = await ex(batch, { pooling: 'mean', normalize: true }); const dim = t.dims[1];
    for (let k = 0; k < batch.length; k++) out.push(Array.from(t.data.slice(k * dim, (k + 1) * dim)));
    onProgress(Math.min(texts.length, i + 32)); await new Promise(r => setTimeout(r, 0));
  }
  return out;
}

let lr: Promise<{ w: number[]; b: number }> | null = null;
export async function positiveProb(emb: number[][]): Promise<number[]> {
  const m = await (lr ??= fetch('/sentiment-lr.json').then(r => r.json()));
  return emb.map(x => { let z = m.b; for (let i = 0; i < x.length; i++) z += m.w[i] * x[i]; return 1 / (1 + Math.exp(-z)); });
}
