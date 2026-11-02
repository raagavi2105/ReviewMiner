import { chooseK, kmeans } from './cluster';

export interface Review { id: number; text: string; rating?: number; pos: number; cluster: number }
export interface Theme {
  id: number; name: string; size: number; share: number; negShare: number; posShare: number; avgPos: number;
  cohesion: number; keywords: string[]; quotes: string[]; reach: number; impact: number; confidence: number; effort: number; score: number;
}
const stop = new Set('the and was were are for with that this have had has but not you your they them its our out very just from there their been would could when what which about all can one will get got also than then too she her him his did does here only really much even still into over more some other like were because after before while those these where being very'.split(' '));
const toks = (s: string) => (s.toLowerCase().match(/[a-z][a-z']{2,}/g) ?? []).filter(w => !stop.has(w));

export function analyze(texts: string[], emb: number[][], pos: number[], ratings: (number | undefined)[], k?: number) {
  const auto = chooseK(emb, 4, 8);
  const kk = k ?? auto.k;
  const fit = k && k !== auto.k ? kmeans(emb, kk, 1, 8) : auto.fit;
  const n = texts.length;
  const reviews: Review[] = texts.map((t, i) => ({ id: i, text: t, rating: ratings[i], pos: pos[i], cluster: fit.labels[i] }));
  // c-TF-IDF keywords
  const docs = Array.from({ length: kk }, () => new Map<string, number>()); const total = new Map<string, number>();
  reviews.forEach(r => toks(r.text).forEach(w => { docs[r.cluster].set(w, (docs[r.cluster].get(w) ?? 0) + 1); total.set(w, (total.get(w) ?? 0) + 1); }));
  const avgLen = [...total.values()].reduce((a, b) => a + b, 0) / kk;
  const themes: Theme[] = [];
  for (let c = 0; c < kk; c++) {
    const members = reviews.filter(r => r.cluster === c); if (!members.length) continue;
    const kw = [...docs[c].entries()].filter(([w, f]) => f >= 2 && (total.get(w) ?? 0) >= 3)
      .map(([w, f]) => [w, (f / members.length) * Math.log(1 + avgLen / (total.get(w) ?? 1))] as [string, number]).sort((a, b) => b[1] - a[1]).slice(0, 6).map(x => x[0]);
    const cent = fit.centroids[c]; const sim = (i: number) => emb[i].reduce((s, v, j) => s + v * cent[j], 0) / (Math.sqrt(cent.reduce((s, v) => s + v * v, 0)) || 1);
    const ranked = members.map(m => ({ m, s: sim(m.id) })).sort((a, b) => b.s - a.s);
    const neg = members.filter(m => m.pos < 0.35).length / members.length, posS = members.filter(m => m.pos > 0.65).length / members.length;
    const cohesion = ranked.reduce((s, x) => s + x.s, 0) / ranked.length;
    const mixed = members.filter(m => m.pos >= 0.35 && m.pos <= 0.65).length / members.length;
    const reach = (members.length * (neg + 0.5 * mixed)) / n, impact = 1 + 2 * neg, confidence = Math.min(1, Math.max(0.5, 0.5 + (cohesion - 0.3) * 1.2));
    const quotes = [...ranked.filter(x => x.m.pos < 0.35).slice(0, 2), ...ranked.slice(0, 3)].map(x => x.m.text).filter((q, i, a) => a.indexOf(q) === i).slice(0, 3);
    themes.push({ id: c, name: kw.slice(0, 2).join(' · ') || `Theme ${c + 1}`, size: members.length, share: members.length / n, negShare: neg, posShare: posS, avgPos: members.reduce((s, m) => s + m.pos, 0) / members.length, cohesion, keywords: kw, quotes, reach, impact, confidence, effort: 3, score: 0 });
  }
  return { reviews, themes: rescore(themes), k: kk, autoK: auto.k, silhouette: auto.score };
}
export const rescore = (themes: Theme[]) => {
  const raw = themes.map(t => (t.reach * t.impact * t.confidence) / t.effort); const mx = Math.max(...raw, 1e-9);
  return themes.map((t, i) => ({ ...t, score: (raw[i] / mx) * 100 })).sort((a, b) => b.score - a.score); // normalized: top theme = 100
};
