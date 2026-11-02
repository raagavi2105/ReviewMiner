// k-means (k-means++ init, multiple restarts) and clustering metrics. Pure TS: used by the app and the eval.
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const d2 = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - b[i]; s += x * x; } return s; };

export function kmeans(X: number[][], k: number, seed = 1, restarts = 6, iters = 60) {
  const n = X.length, dim = X[0].length;
  let best = { labels: new Array<number>(n).fill(0), centroids: [] as number[][], inertia: Infinity };
  for (let rs = 0; rs < restarts; rs++) {
    const rand = rng(seed * 1000 + rs);
    const cents: number[][] = [X[Math.floor(rand() * n)].slice()];
    while (cents.length < k) {
      const dist = X.map(x => Math.min(...cents.map(c => d2(x, c))));
      const tot = dist.reduce((a, b) => a + b, 0); let r = rand() * tot, idx = 0;
      for (; idx < n - 1; idx++) { r -= dist[idx]; if (r <= 0) break; }
      cents.push(X[idx].slice());
    }
    let labels = new Array<number>(n).fill(-1);
    for (let it = 0; it < iters; it++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let bi = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { const dd = d2(X[i], cents[c]); if (dd < bd) { bd = dd; bi = c; } }
        if (labels[i] !== bi) { labels[i] = bi; changed = true; }
      }
      const sums = Array.from({ length: k }, () => new Array(dim).fill(0)); const cnt = new Array(k).fill(0);
      for (let i = 0; i < n; i++) { cnt[labels[i]]++; for (let j = 0; j < dim; j++) sums[labels[i]][j] += X[i][j]; }
      for (let c = 0; c < k; c++) if (cnt[c]) for (let j = 0; j < dim; j++) cents[c][j] = sums[c][j] / cnt[c];
      if (!changed) break;
    }
    let inertia = 0; for (let i = 0; i < n; i++) inertia += d2(X[i], cents[labels[i]]);
    if (inertia < best.inertia) best = { labels: labels.slice(), centroids: cents.map(c => c.slice()), inertia };
  }
  return best;
}

export function silhouette(X: number[][], labels: number[], sample = 400, seed = 3): number {
  const n = X.length, rand = rng(seed);
  const idx = n <= sample ? [...Array(n).keys()] : Array.from({ length: sample }, () => Math.floor(rand() * n));
  const k = Math.max(...labels) + 1; let tot = 0, cnt = 0;
  for (const i of idx) {
    const sum = new Array(k).fill(0), c = new Array(k).fill(0);
    for (let j = 0; j < n; j++) if (j !== i) { sum[labels[j]] += Math.sqrt(d2(X[i], X[j])); c[labels[j]]++; }
    const own = labels[i]; if (!c[own]) continue;
    const a = sum[own] / c[own]; let b = Infinity;
    for (let q = 0; q < k; q++) if (q !== own && c[q]) b = Math.min(b, sum[q] / c[q]);
    if (isFinite(b)) { tot += (b - a) / Math.max(a, b); cnt++; }
  }
  return cnt ? tot / cnt : 0;
}

export function chooseK(X: number[][], kmin = 3, kmax = 8, seed = 1) {
  let best = { k: kmin, score: -2, fit: kmeans(X, kmin, seed) };
  for (let k = kmin; k <= Math.min(kmax, X.length - 1); k++) {
    const fit = kmeans(X, k, seed, 4); const score = silhouette(X, fit.labels);
    if (score > best.score) best = { k, score, fit };
  }
  return best;
}

// ---- external metrics vs ground truth ----
function contingency(a: number[], b: number[]) {
  const ka = Math.max(...a) + 1, kb = Math.max(...b) + 1;
  const m = Array.from({ length: ka }, () => new Array(kb).fill(0));
  a.forEach((x, i) => m[x][b[i]]++); return m;
}
export function purity(pred: number[], truth: number[]) {
  const m = contingency(pred, truth); return m.reduce((s, r) => s + Math.max(...r), 0) / pred.length;
}
export function nmi(pred: number[], truth: number[]) {
  const m = contingency(pred, truth), n = pred.length;
  const ra = m.map(r => r.reduce((a, b) => a + b, 0)), cb = m[0].map((_, j) => m.reduce((s, r) => s + r[j], 0));
  let mi = 0; m.forEach((r, i) => r.forEach((v, j) => { if (v) mi += (v / n) * Math.log((v * n) / (ra[i] * cb[j])); }));
  const h = (v: number[]) => -v.reduce((s, x) => (x ? s + (x / n) * Math.log(x / n) : s), 0);
  return mi / Math.sqrt(h(ra) * h(cb));
}
export function ari(pred: number[], truth: number[]) {
  const m = contingency(pred, truth), n = pred.length, c2 = (x: number) => (x * (x - 1)) / 2;
  const sumIj = m.flat().reduce((s, v) => s + c2(v), 0);
  const sumA = m.reduce((s, r) => s + c2(r.reduce((a, b) => a + b, 0)), 0);
  const sumB = m[0].reduce((s, _, j) => s + c2(m.reduce((a, r) => a + r[j], 0)), 0);
  const exp = (sumA * sumB) / c2(n), mx = (sumA + sumB) / 2;
  return (sumIj - exp) / (mx - exp);
}
