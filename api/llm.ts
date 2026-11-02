// Vercel serverless proxy to Groq. The key lives only in the GROQ_API_KEY environment variable.
import type { VercelRequest, VercelResponse } from '@vercel/node';
const MODEL = process.env.GROQ_MODEL || 'qwen/qwen3.8-27b';

const PROMPTS: Record<string, { system: string; max: number }> = {
  name: { max: 700, system: 'You label customer-feedback clusters. For each cluster (keywords + sample reviews) return a short product-theme name (2-4 words, Title Case, no generic words like "Feedback"). Reply with ONLY a JSON array of strings, one per cluster, in order.' },
  prd: { max: 1400, system: `You are a Group Product Manager writing a crisp one-page PRD from customer evidence. Use ONLY the facts given; never invent numbers. Format in Markdown with these sections: ## Problem, ## Who is affected, ## Evidence (quote 2-3 real reviews), ## Goals & success metrics (include a north-star and 2 guardrails), ## Proposed solution (3-5 bullets), ## Risks & open questions, ## Rollout plan. Keep it under 350 words.` },
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const key = process.env.GROQ_API_KEY; if (!key) return res.status(500).json({ error: 'Server is missing GROQ_API_KEY' });
  const { task, input } = req.body ?? {}; const p = PROMPTS[task as string];
  if (!p) return res.status(400).json({ error: 'Unknown task' });
  const body = JSON.stringify(input ?? {}); if (body.length > 12000) return res.status(413).json({ error: 'Input too large' });
  const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: MODEL, temperature: 0.3, max_tokens: p.max, ...(MODEL.includes('gpt-oss') ? { reasoning_effort: 'low' } : {}), messages: [{ role: 'system', content: p.system }, { role: 'user', content: body }] }),
  });
  if (!r.ok) return res.status(502).json({ error: 'Model error' });
  const d = await r.json(); return res.status(200).json({ text: d.choices?.[0]?.message?.content ?? '', model: MODEL });
}
