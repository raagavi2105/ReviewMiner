// Filters SemEval-2014 restaurant sentences to single-theme, clear-polarity rows and embeds them with
// the same model the browser uses (all-MiniLM-L6-v2, quantized, via transformers.js).
import { readFileSync, writeFileSync } from 'node:fs';
import { pipeline } from '@xenova/transformers';

const THEMES = ['food', 'service', 'ambience', 'price'];
function load(split: string) {
  const raw = JSON.parse(readFileSync(`data/restaurants_${split}.json`, 'utf8'));
  const rows: { text: string; theme: string; polarity: string }[] = [];
  for (const r of raw) {
    const cats = r.aspectCategories as { category: string; polarity: string }[];
    const uniq = [...new Set(cats.map(c => c.category))].filter(c => THEMES.includes(c));
    if (uniq.length !== 1 || cats.some(c => !THEMES.includes(c.category) && c.category !== 'anecdotes/miscellaneous')) continue;
    const pols = [...new Set(cats.filter(c => c.category === uniq[0]).map(c => c.polarity))];
    if (pols.length !== 1 || !['positive', 'negative'].includes(pols[0])) continue;
    rows.push({ text: r.text, theme: uniq[0], polarity: pols[0] });
  }
  return rows;
}
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
for (const split of ['train', 'test']) {
  const rows = load(split);
  const emb: number[][] = [];
  for (let i = 0; i < rows.length; i += 64) {
    const batch = rows.slice(i, i + 64).map(r => r.text);
    const t = await extractor(batch, { pooling: 'mean', normalize: true });
    const dim = t.dims[1];
    for (let k = 0; k < batch.length; k++) emb.push(Array.from(t.data.slice(k * dim, (k + 1) * dim) as Float32Array).map(v => +v.toFixed(5)));
  }
  writeFileSync(`data/${split}.json`, JSON.stringify({ rows, emb }));
  console.log(split, rows.length, 'themes', Object.fromEntries(THEMES.map(t => [t, rows.filter(r => r.theme === t).length])));
}
