import { useMemo, useRef, useState } from 'react';
import { detect, toCsv } from './lib/csv';
import { embed, loadEmbedder, positiveProb } from './lib/ml';
import { analyze, rescore, type Review, type Theme } from './lib/analysis';
import clusterEval from './eval/eval-clusters.json';
import sentEval from './eval/eval-sentiment-lr.json';
import llmEval from './eval/eval-llm.json';

const COLORS = ['#6d5efc', '#ec4899', '#10b981', '#f59e0b', '#06b6d4', '#ef4444', '#8b5cf6', '#84cc16'];
const pct = (x: number, d = 0) => (x * 100).toFixed(d) + '%';
const sentLabel = (p: number) => (p < 0.35 ? 'neg' : p > 0.65 ? 'pos' : 'mix');
const sentText = { neg: 'Negative', pos: 'Positive', mix: 'Mixed' } as const;

async function llm(task: string, input: unknown): Promise<string> {
  const r = await fetch('/api/llm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task, input }) });
  const d = await r.json(); if (!r.ok) throw new Error(d.error || 'Request failed'); return d.text;
}

function Md({ text }: { text: string }) {
  const esc = (x: string) => x.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const inline = (x: string) => esc(x).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  const html = text.split('\n').map(l => {
    if (/^#{1,3}\s/.test(l)) return `<h2>${inline(l.replace(/^#{1,3}\s/, ''))}</h2>`;
    if (/^>\s?/.test(l)) return `<blockquote>${inline(l.replace(/^>\s?/, ''))}</blockquote>`;
    if (/^\s*([-*]|\d+\.)\s/.test(l)) return `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ''))}</li>`;
    return l.trim() ? `<p>${inline(l)}</p>` : '';
  }).join('');
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />;
}

type Stage = 'upload' | 'processing' | 'results';
interface Step { label: string; state: 'wait' | 'act' | 'done'; detail?: string; pct?: number }

export default function App() {
  const [stage, setStage] = useState<Stage>('upload');
  const [steps, setSteps] = useState<Step[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [themes, setThemes] = useState<Theme[]>([]);
  const [meta, setMeta] = useState({ k: 0, autoK: 0, name: '' });
  const [error, setError] = useState('');
  const [over, setOver] = useState(false);
  const [paste, setPaste] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);
  const cache = useRef<{ texts: string[]; emb: number[][]; pos: number[]; ratings: (number | undefined)[] } | null>(null);

  const upd = (i: number, p: Partial<Step>) => setSteps(s => s.map((x, j) => (j === i ? { ...x, ...p } : j < i ? { ...x, state: 'done' } : x)));

  async function run(csv: string, name: string) {
    setError('');
    const parsed = csv.includes(',') || csv.includes('\n') ? detect(csv.includes('\n') ? csv : csv) : detect(csv);
    let texts = parsed.rows.map(r => (r[parsed.textCol] ?? '').trim()).filter(t => t.length > 8);
    let ratings: (number | undefined)[] = parsed.rows.filter(r => (r[parsed.textCol] ?? '').trim().length > 8).map(r => (parsed.ratingCol >= 0 ? Number(r[parsed.ratingCol]) || undefined : undefined));
    if (texts.length < 12) { setError('I need at least 12 reviews (one per row, with a header like "review"). Try the sample data to see how it works.'); return; }
    if (texts.length > 1500) { texts = texts.slice(0, 1500); ratings = ratings.slice(0, 1500); }
    setStage('processing');
    setSteps([{ label: 'Loading the language model (one-time download, ~23 MB)', state: 'act' }, { label: `Understanding ${texts.length.toLocaleString()} reviews`, state: 'wait' }, { label: 'Finding themes and scoring sentiment', state: 'wait' }, { label: 'Naming themes with AI', state: 'wait' }]);
    try {
      await loadEmbedder(p => upd(0, { pct: p / 100, detail: `${Math.round(p)}%` }));
      upd(1, { state: 'act' });
      const emb = await embed(texts, d => upd(1, { pct: d / texts.length, detail: `${d} / ${texts.length}` }));
      upd(2, { state: 'act' }); await new Promise(r => setTimeout(r, 30));
      const pos = await positiveProb(emb);
      cache.current = { texts, emb, pos, ratings };
      const a = analyze(texts, emb, pos, ratings);
      upd(3, { state: 'act' });
      let named = a.themes;
      try {
        const out = await llm('name', a.themes.map(t => ({ keywords: t.keywords, reviews: t.quotes })));
        const arr = JSON.parse(out.slice(out.indexOf('['), out.lastIndexOf(']') + 1)) as string[];
        const byId = new Map(a.themes.map((t, i) => [t.id, arr[i]])); named = a.themes.map(t => ({ ...t, name: byId.get(t.id) || t.name }));
      } catch { /* keep keyword names */ }
      upd(3, { state: 'done' });
      setReviews(a.reviews); setThemes(named); setMeta({ k: a.k, autoK: a.autoK, name }); setStage('results');
    } catch (e) { setError('Something went wrong while analysing: ' + (e as Error).message); setStage('upload'); }
  }
  const onFile = (f: File) => f.text().then(t => run(t, f.name));
  const sample = () => fetch('/demo-reviews.csv').then(r => r.text()).then(t => run(t, 'Sample: restaurant reviews'));

  async function regroup(k: number) {
    if (!cache.current) return; const c = cache.current;
    const a = analyze(c.texts, c.emb, c.pos, c.ratings, k); setReviews(a.reviews); setThemes(a.themes); setMeta(m => ({ ...m, k: a.k }));
    try {
      const out = await llm('name', a.themes.map(t => ({ keywords: t.keywords, reviews: t.quotes })));
      const arr = JSON.parse(out.slice(out.indexOf('['), out.lastIndexOf(']') + 1)) as string[];
      setThemes(a.themes.map((t, i) => ({ ...t, name: arr[i] || t.name })));
    } catch { /* ignore */ }
  }

  return (
    <div className="wrap">
      <div className="nav">
        <div className="logo"><i>◎</i> ReviewMiner</div>
        {stage === 'results' && <button className="btn sm" onClick={() => { setStage('upload'); setPaste(''); }}>＋ New analysis</button>}
      </div>
      {stage === 'upload' && (
        <div className="fade">
          <div className="hero">
            <h1>Turn thousands of reviews into a <span className="serif" style={{ background: 'linear-gradient(90deg,var(--ac),var(--ac2))', WebkitBackgroundClip: 'text', color: 'transparent' }}>prioritized roadmap</span></h1>
            <p>Drop in customer reviews. ReviewMiner finds the themes, measures how people feel, ranks what to fix first, and drafts the PRD.</p>
          </div>
          <div className={`drop ${over ? 'over' : ''}`} onClick={() => fileRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={e => { e.preventDefault(); setOver(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); }}>
            <div style={{ fontSize: 34 }}>📄</div><b>Drop a CSV here, or click to browse</b><span>One review per row, with a header such as “review” or “text”. Up to 1,500 reviews.</span>
            <input ref={fileRef} type="file" accept=".csv,.txt" hidden onChange={e => e.target.files?.[0] && onFile(e.target.files[0])} />
          </div>
          {error && <p style={{ color: 'var(--bad)' }}>{error}</p>}
          <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
            <button className="btn pri" onClick={sample}>✨ Try with sample data</button>
          </div>
          <details style={{ marginTop: 18, textAlign: 'center' }}>
            <summary className="mut sm" style={{ cursor: 'pointer' }}>…or paste reviews (one per line)</summary>
            <textarea rows={6} value={paste} onChange={e => setPaste(e.target.value)} placeholder="The app crashes every time I open settings&#10;Love the new dark mode…" style={{ width: '100%', marginTop: 10 }} />
            <button className="btn sm" style={{ marginTop: 8 }} disabled={paste.trim().split('\n').length < 12} onClick={() => run('review\n' + paste.trim().split('\n').map(l => `"${l.replace(/"/g, '""')}"`).join('\n'), 'Pasted reviews')}>Analyze pasted reviews</button>
          </details>
          <div className="chips"><span className="chip">🔒 Embeddings run in your browser</span><span className="chip">🧠 MiniLM + k-means + logistic regression</span><span className="chip">📊 Validated on human-labeled data</span></div>
        </div>
      )}
      {stage === 'processing' && (
        <div className="card fade" style={{ maxWidth: 620, margin: '40px auto' }}>
          <h2>Analyzing your reviews</h2><p className="mut sm">Everything except theme naming happens on your device.</p>
          {steps.map((s, i) => (
            <div className="step" key={i}><div className={`dot ${s.state === 'done' ? 'done' : s.state === 'act' ? 'act' : ''}`}>{s.state === 'done' ? '✓' : i + 1}</div>
              <div style={{ flex: 1 }}><div style={{ fontWeight: 600, color: s.state === 'wait' ? 'var(--mut)' : 'var(--tx)' }}>{s.label} <span className="mut sm">{s.detail}</span></div>
                {s.state === 'act' && s.pct !== undefined && <div className="prog"><div style={{ width: `${s.pct * 100}%` }} /></div>}</div></div>
          ))}
        </div>
      )}
      {stage === 'results' && <Results reviews={reviews} themes={themes} setThemes={setThemes} meta={meta} regroup={regroup} />}
    </div>
  );
}

function SentBar({ neg, pos }: { neg: number; pos: number }) {
  return <div className="bar" title={`${pct(neg)} negative · ${pct(pos)} positive`}><div style={{ width: pct(neg), background: 'var(--bad)' }} /><div style={{ width: pct(1 - neg - pos), background: 'var(--warn)', opacity: .6 }} /><div style={{ width: pct(pos), background: 'var(--good)' }} /></div>;
}

function Results({ reviews, themes, setThemes, meta, regroup }: { reviews: Review[]; themes: Theme[]; setThemes: (t: Theme[]) => void; meta: { k: number; autoK: number; name: string }; regroup: (k: number) => void }) {
  const [tab, setTab] = useState('overview');
  const [sel, setSel] = useState<number>(themes[0]?.id ?? 0);
  const color = (id: number) => COLORS[id % COLORS.length];
  const negAll = reviews.filter(r => r.pos < 0.35).length / reviews.length, posAll = reviews.filter(r => r.pos > 0.65).length / reviews.length;
  const top = themes[0];
  const tabs: [string, string][] = [['overview', 'Overview'], ['themes', 'Themes'], ['prioritize', 'Prioritize'], ['reviews', 'Reviews'], ['prd', 'PRD'], ['lab', 'Model lab']];
  const go = (t: string, id?: number) => { setTab(t); if (id !== undefined) setSel(id); };
  const setEffort = (id: number, effort: number) => setThemes(rescore(themes.map(t => (t.id === id ? { ...t, effort } : t))));
  const rename = (id: number, name: string) => setThemes(themes.map(t => (t.id === id ? { ...t, name } : t)));

  return (
    <div className="fade">
      <div className="row between"><div><h2 style={{ fontSize: 26 }}>{meta.name}</h2><span className="mut sm">{reviews.length.toLocaleString()} reviews · {themes.length} themes</span></div>
        <div className="row"><span className="mut sm">Themes</span><input type="range" min={3} max={8} value={meta.k} onChange={e => regroup(Number(e.target.value))} style={{ padding: 0 }} /><b>{meta.k}</b></div></div>
      <div className="tabs">{tabs.map(([k, l]) => <button key={k} className={`tab ${tab === k ? 'on' : ''}`} onClick={() => setTab(k)}>{l}</button>)}</div>

      {tab === 'overview' && (<div className="grid">
        <div className="grid g4">
          <div className="card kpi"><b>{reviews.length.toLocaleString()}</b><span>reviews analyzed</span></div>
          <div className="card kpi"><b>{themes.length}</b><span>themes discovered</span></div>
          <div className="card kpi"><b style={{ color: 'var(--bad)' }}>{pct(negAll)}</b><span>negative sentiment</span></div>
          <div className="card kpi"><b style={{ color: 'var(--good)' }}>{pct(posAll)}</b><span>positive sentiment</span></div>
        </div>
        {top && <div className="card" style={{ background: 'linear-gradient(135deg,rgba(109,94,252,.12),rgba(236,72,153,.1))' }}>
          <span className="pill mix">#1 PRIORITY</span><h2 style={{ marginTop: 10 }}>{top.name}</h2>
          <p className="mut" style={{ margin: '4px 0 12px' }}>{pct(top.share)} of reviews are about this, and {pct(top.negShare)} of those are negative, so about {pct(top.reach)} of all reviews voice a problem here. RICE score {top.score.toFixed(0)}.</p>
          {top.quotes[0] && <div className="quote">“{top.quotes[0]}”</div>}
          <div className="row" style={{ marginTop: 12 }}><button className="btn pri" onClick={() => go('prd', top.id)}>Draft PRD →</button><button className="btn" onClick={() => go('prioritize')}>See ranking</button></div></div>}
        <div className="card"><h2>What people talk about</h2><p className="mut sm" style={{ marginBottom: 14 }}>Share of reviews per theme, colored by sentiment.</p>
          {themes.map(t => <div key={t.id} style={{ margin: '12px 0', cursor: 'pointer' }} onClick={() => go('themes', t.id)}>
            <div className="row between sm"><b><span style={{ color: color(t.id) }}>●</span> {t.name}</b><span className="mut">{pct(t.share)} · {t.size} reviews</span></div>
            <div style={{ marginTop: 6 }}><SentBar neg={t.negShare} pos={t.posShare} /></div></div>)}
          <p className="mut sm">Red = negative · Amber = mixed · Green = positive</p></div>
      </div>)}

      {tab === 'themes' && <div className="grid g3">{themes.map(t => (
        <div className="card" key={t.id} style={{ borderTop: `3px solid ${color(t.id)}` }}>
          <input className="name" value={t.name} onChange={e => rename(t.id, e.target.value)} aria-label="Theme name" />
          <div className="row sm mut" style={{ margin: '4px 0 12px' }}><span>{t.size} reviews</span>·<span>{pct(t.share)} of total</span>·<span className={`pill ${t.negShare > .5 ? 'neg' : t.negShare > .25 ? 'mix' : 'pos'}`}>{pct(t.negShare)} negative</span></div>
          <SentBar neg={t.negShare} pos={t.posShare} />
          <div style={{ margin: '12px 0' }}>{t.keywords.map(k => <span className="kw" key={k}>{k}</span>)}</div>
          {t.quotes.map((q, i) => <div className="quote" key={i}>“{q}”</div>)}
          <button className="btn sm" style={{ marginTop: 8 }} onClick={() => go('prd', t.id)}>Draft PRD</button></div>))}</div>}

      {tab === 'prioritize' && (<div className="grid">
        <div className="card"><h2>Impact vs. reach</h2><p className="mut sm">Reach = share of all reviews voicing a problem in that theme. Top-right is where to focus. Bubble size = RICE score.</p><Bubble themes={themes} color={color} onPick={id => go('themes', id)} /></div>
        <div className="card" style={{ overflow: 'auto' }}><div className="row between"><div><h2>RICE ranking</h2><p className="mut sm">Reach × Impact × Confidence ÷ Effort, scaled so the top theme is 100. Adjust effort to re-rank live.</p></div>
          <button className="btn sm" onClick={() => { const b = new Blob([toCsv([['rank', 'theme', 'reviews', 'reach_pct', 'impact', 'confidence', 'effort', 'rice'], ...themes.map((t, i) => [i + 1, t.name, t.size, (t.reach * 100).toFixed(1), t.impact.toFixed(2), t.confidence.toFixed(2), t.effort, t.score.toFixed(1)])])], { type: 'text/csv' }); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'rice-ranking.csv'; a.click(); }}>Export CSV</button></div>
          <table><thead><tr><th>#</th><th>Theme</th><th>Reach</th><th>Impact</th><th>Confidence</th><th>Effort</th><th>RICE</th></tr></thead><tbody>
            {themes.map((t, i) => <tr key={t.id}><td>{i + 1}</td><td><b>{t.name}</b></td><td>{pct(t.reach)}</td><td>{t.impact.toFixed(1)} / 3</td><td>{pct(t.confidence)}</td>
              <td><select value={t.effort} onChange={e => setEffort(t.id, Number(e.target.value))}>{[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{['', 'XS', 'S', 'M', 'L', 'XL'][n]}</option>)}</select></td><td><b>{t.score.toFixed(0)}</b></td></tr>)}</tbody></table>
          <p className="mut sm" style={{ marginTop: 10 }}>Impact = 1 + 2 × share of negative reviews. Confidence reflects how tightly the reviews cluster.</p></div>
      </div>)}

      {tab === 'reviews' && <ReviewsTab reviews={reviews} themes={themes} color={color} />}
      {tab === 'prd' && <Prd themes={themes} sel={sel} setSel={setSel} color={color} />}
      {tab === 'lab' && <Lab />}
    </div>
  );
}

function Bubble({ themes, color, onPick }: { themes: Theme[]; color: (i: number) => string; onPick: (i: number) => void }) {
  const W = 640, H = 300, P = 36; const maxR = Math.max(...themes.map(t => t.reach), 0.01) * 1.25; const maxS = Math.max(...themes.map(t => t.score), 1);
  return (<svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" aria-label="Bubble chart of impact versus reach">
    <rect x={P} y={10} width={W - P - 10} height={H - P - 10} rx={12} fill="var(--surf2)" opacity=".5" />
    <line x1={P + (W - P - 10) / 2} x2={P + (W - P - 10) / 2} y1={10} y2={H - P} stroke="var(--line)" strokeDasharray="4" /><line x1={P} x2={W - 10} y1={(H - P + 10) / 2} y2={(H - P + 10) / 2} stroke="var(--line)" strokeDasharray="4" />
    <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="var(--mut)">Reach (share of reviews with a problem) →</text>
    <text x={12} y={H / 2} fontSize="11" fill="var(--mut)" transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle">Impact →</text>
    {themes.map(t => { const x = P + (t.reach / maxR) * (W - P - 30) + 10, y = H - P - ((t.impact - 1) / 2) * (H - P - 50) - 14, r = 14 + 26 * Math.sqrt(t.score / maxS);
      return <g key={t.id} style={{ cursor: 'pointer' }} onClick={() => onPick(t.id)}><circle cx={x} cy={y} r={r} fill={color(t.id)} opacity=".25" stroke={color(t.id)} strokeWidth="2" /><text x={x} y={y + 4} textAnchor="middle" fontSize="11" fontWeight="600" fill="var(--tx)">{t.name.split(' ')[0].slice(0, 12)}</text></g>; })}
  </svg>);
}

function ReviewsTab({ reviews, themes, color }: { reviews: Review[]; themes: Theme[]; color: (i: number) => string }) {
  const [q, setQ] = useState(''); const [th, setTh] = useState('all'); const [se, setSe] = useState('all');
  const list = useMemo(() => reviews.filter(r => (th === 'all' || r.cluster === Number(th)) && (se === 'all' || sentLabel(r.pos) === se) && r.text.toLowerCase().includes(q.toLowerCase())), [reviews, q, th, se]);
  const name = (id: number) => themes.find(t => t.id === id)?.name ?? '';
  return (<div className="card"><div className="row" style={{ marginBottom: 12 }}>
    <input placeholder="Search reviews…" value={q} onChange={e => setQ(e.target.value)} style={{ flex: 1, minWidth: 160 }} />
    <select value={th} onChange={e => setTh(e.target.value)}><option value="all">All themes</option>{themes.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</select>
    <select value={se} onChange={e => setSe(e.target.value)}><option value="all">All sentiment</option><option value="neg">Negative</option><option value="mix">Mixed</option><option value="pos">Positive</option></select></div>
    <p className="mut sm">{list.length.toLocaleString()} matching · showing first 100</p>
    <table><thead><tr><th>Review</th><th>Theme</th><th>Sentiment</th></tr></thead><tbody>{list.slice(0, 100).map(r => { const s = sentLabel(r.pos); return <tr key={r.id}><td>{r.text}</td><td style={{ whiteSpace: 'nowrap' }}><span style={{ color: color(r.cluster) }}>●</span> {name(r.cluster)}</td><td><span className={`pill ${s}`}>{sentText[s]}</span></td></tr>; })}</tbody></table></div>);
}

function Prd({ themes, sel, setSel, color }: { themes: Theme[]; sel: number; setSel: (n: number) => void; color: (i: number) => string }) {
  const [text, setText] = useState(''); const [busy, setBusy] = useState(false); const [err, setErr] = useState(''); const [cp, setCp] = useState(false);
  const t = themes.find(x => x.id === sel) ?? themes[0];
  async function gen() { setBusy(true); setErr(''); setText(''); try { setText(await llm('prd', { theme: t.name, share_of_reviews: pct(t.share), negative_share: pct(t.negShare), review_count: t.size, keywords: t.keywords, real_reviews: t.quotes, rice_score: Math.round(t.score) })); } catch (e) { setErr((e as Error).message); } setBusy(false); }
  return (<div className="card"><div className="row between"><div><h2>Draft a PRD</h2><p className="mut sm">Grounded only in your reviews. Edit before sharing.</p></div>
    <div className="row"><select value={t.id} onChange={e => { setSel(Number(e.target.value)); setText(''); }}>{themes.map(x => <option key={x.id} value={x.id}>{x.name}</option>)}</select><button className="btn pri" disabled={busy} onClick={gen}>{busy ? 'Writing…' : '✨ Generate PRD'}</button></div></div>
    <div className="mut sm" style={{ marginTop: 8 }}><span style={{ color: color(t.id) }}>●</span> {t.size} reviews · {pct(t.negShare)} negative</div>
    {err && <p style={{ color: 'var(--bad)' }}>{err}</p>}
    {text && <><div style={{ margin: '16px 0', padding: 18, background: 'var(--surf2)', borderRadius: 12 }}><Md text={text} /></div>
      <div className="row"><button className="btn sm" onClick={() => { navigator.clipboard.writeText(text); setCp(true); setTimeout(() => setCp(false), 1500); }}>{cp ? '✓ Copied' : 'Copy Markdown'}</button>
        <button className="btn sm" onClick={() => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/markdown' })); a.download = `prd-${t.name.replace(/\W+/g, '-').toLowerCase()}.md`; a.click(); }}>Download .md</button></div></>}
  </div>);
}

function Lab() {
  const llmRows = Object.entries(llmEval as Record<string, { n: number; accuracy: number; avgLatencyMs: number; failedCalls: number }>);
  return (<div className="grid">
    <div className="card"><h2>How accurate is this?</h2><p className="mut sm">Measured on SemEval-2014 restaurant sentences with human-labeled themes and sentiment (not on your data). Reproduce with <code>npm run eval</code>.</p></div>
    <div className="grid g2">
      <div className="card"><h3>Theme discovery (k-means, k = 4)</h3><p className="mut sm">{clusterEval.n.toLocaleString()} sentences, 4 human themes. Majority-class baseline: 55%.</p>
        <table><thead><tr><th>Method</th><th>Purity</th><th>NMI</th></tr></thead><tbody>
          <tr><td><b>MiniLM embeddings</b></td><td><b>{pct(clusterEval.embeddingsKMeans_k4.purity, 1)}</b></td><td>{clusterEval.embeddingsKMeans_k4.nmi.toFixed(2)}</td></tr>
          <tr><td>TF-IDF baseline</td><td>{pct(clusterEval.tfidfKMeans_k4.purity, 1)}</td><td>{clusterEval.tfidfKMeans_k4.nmi.toFixed(2)}</td></tr></tbody></table>
        <p className="mut sm" style={{ marginTop: 8 }}>Auto-choosing k by silhouette is weak ({clusterEval.autoK.chosenK} chosen), so use the Themes slider.</p></div>
      <div className="card"><h3>Sentiment classifier</h3><p className="mut sm">Logistic regression on embeddings, trained on {sentEval.n_train.toLocaleString()} sentences, tested on {sentEval.n_test}. Runs in your browser.</p>
        <table><thead><tr><th>Model</th><th>Accuracy</th><th>Latency</th></tr></thead><tbody>
          <tr><td><b>Embeddings + LR (in browser)</b></td><td><b>{pct(sentEval.lr_accuracy, 1)}</b></td><td>~1 ms</td></tr>
          <tr><td>Majority-class</td><td>{pct(sentEval.majority_baseline_acc, 1)}</td><td>–</td></tr>
          {llmRows.map(([m, r]) => <tr key={m}><td>{m} (zero-shot LLM)</td><td>{pct(r.accuracy, 1)}</td><td>{r.avgLatencyMs} ms</td></tr>)}</tbody></table></div>
    </div></div>);
}
