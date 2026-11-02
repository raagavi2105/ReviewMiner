// Theme discovery evaluation against human aspect labels (SemEval-2014 Task 4, restaurants).
import { readFileSync, writeFileSync } from 'node:fs';
import { kmeans, chooseK, nmi, ari, purity } from '../src/lib/cluster';
const tr = JSON.parse(readFileSync('data/train.json', 'utf8')), te = JSON.parse(readFileSync('data/test.json', 'utf8'));
const rows = [...tr.rows, ...te.rows] as { text: string; theme: string }[];
const X = [...tr.emb, ...te.emb] as number[][];
const themes = ['food', 'service', 'ambience', 'price'];
const truth = rows.map(r => themes.indexOf(r.theme));

// TF-IDF baseline (same k-means)
const tok = (s: string) => s.toLowerCase().match(/[a-z']{3,}/g) ?? [];
const stop = new Set('the and was were are for with that this have had has but not you your they them its our out very just from there their been would could when what which about all can one will get got also than then too she her him his had did does'.split(' '));
const docs = rows.map(r => tok(r.text).filter(w => !stop.has(w)));
const df = new Map<string, number>(); docs.forEach(d => new Set(d).forEach(w => df.set(w, (df.get(w) ?? 0) + 1)));
const vocab = [...df.entries()].filter(([, c]) => c >= 3).sort((a, b) => b[1] - a[1]).slice(0, 1500).map(([w]) => w);
const vi = new Map(vocab.map((w, i) => [w, i]));
const T = docs.map(d => { const v = new Array(vocab.length).fill(0); d.forEach(w => { const i = vi.get(w); if (i !== undefined) v[i]++; });
  const n = Math.sqrt(v.reduce((s, x, i) => s + (x * Math.log(docs.length / df.get(vocab[i])!)) ** 2, 0)) || 1;
  return v.map((x, i) => (x * Math.log(docs.length / df.get(vocab[i])!)) / n); });

const emb = kmeans(X, 4, 1, 10), tf = kmeans(T, 4, 1, 10), auto = chooseK(X, 3, 8);
const out = {
  dataset: 'SemEval-2014 Task 4 restaurants, single-theme sentences', n: rows.length, humanThemes: themes,
  embeddingsKMeans_k4: { nmi: nmi(emb.labels, truth), ari: ari(emb.labels, truth), purity: purity(emb.labels, truth) },
  tfidfKMeans_k4: { nmi: nmi(tf.labels, truth), ari: ari(tf.labels, truth), purity: purity(tf.labels, truth) },
  autoK: { chosenK: auto.k, silhouette: auto.score, nmi: nmi(auto.fit.labels, truth), purity: purity(auto.fit.labels, truth) },
};
writeFileSync('eval-clusters.json', JSON.stringify(out, null, 2)); console.log(JSON.stringify(out, null, 2));
