// Zero-shot LLM baseline for sentiment + model comparison. Reads GROQ_API_KEY from env.
import { readFileSync, writeFileSync } from 'node:fs';
const key = process.env.GROQ_API_KEY!;
const te = JSON.parse(readFileSync('data/test.json', 'utf8')).rows as { text: string; polarity: string }[];
const N = 80;
const sample = te.filter((_, i) => i % Math.floor(te.length / N) === 0).slice(0, N);
const models = process.argv.slice(2);
const out: Record<string, unknown> = {};
for (const model of models) {
  let correct = 0, fail = 0, ms = 0, tok = 0;
  for (const r of sample) {
    const t0 = Date.now();
    try {
      let res: Response = null as unknown as Response;
      for (let a = 0; a < 8; a++) { res = await fetch('https://api.groq.com/openai/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
        body: JSON.stringify({ model, temperature: 0, max_tokens: 200, ...(model.includes('gpt-oss') ? { reasoning_effort: 'low' } : {}), messages: [{ role: 'system', content: 'Classify the sentiment of the review sentence. Answer with exactly one word: positive or negative.' }, { role: 'user', content: r.text }] }) }); if (res.status !== 429) break; await new Promise(r => setTimeout(r, 2500 * (a + 1))); }
      const d = await res.json(); ms += Date.now() - t0;
      const a = (d.choices?.[0]?.message?.content ?? '').toLowerCase(); tok += d.usage?.total_tokens ?? 0;
      if (!a) fail++; else if (a.includes(r.polarity)) correct++;
    } catch { fail++; }
    await new Promise(r => setTimeout(r, 1200));
  }
  out[model] = { n: sample.length, accuracy: correct / sample.length, failedCalls: fail, avgLatencyMs: Math.round(ms / sample.length), avgTokensPerReview: Math.round(tok / sample.length) };
  console.log(model, out[model]);
}
writeFileSync('eval-llm.json', JSON.stringify(out, null, 2));
