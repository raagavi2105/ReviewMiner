# ReviewMiner

Turn thousands of customer reviews into a prioritized roadmap: it finds themes, measures sentiment, ranks what to fix first with RICE, and drafts a grounded PRD.

**Flow:** upload CSV (or paste / try sample) → themes + sentiment → impact-vs-reach chart and live-editable RICE ranking → searchable reviews → one-click PRD (copy / download Markdown).

## How it works (and what is ML)
1. **Embeddings:** `all-MiniLM-L6-v2` runs **in the browser** (transformers.js), so review text never leaves the device.
2. **Themes:** k-means++ on the embeddings (slider for k; silhouette picks the default), keywords by c-TF-IDF, representative quotes nearest each centroid.
3. **Sentiment:** logistic regression trained on the embeddings and shipped as a 384-weight JSON (runs in ~1 ms/review).
4. **Prioritization:** RICE. Reach = share of all reviews voicing a problem in the theme, Impact = 1 + 2 x negative share, Confidence = cluster cohesion, Effort = your input.
5. **AI writing:** a Vercel serverless function calls Groq (`qwen/qwen3.8-27b`) to name themes and draft the PRD from the numbers and real quotes only.

## Validation (reproducible)
Data: **SemEval-2014 Task 4, restaurants** (human-labeled aspect categories and polarity), single-theme sentences: 1362 train / 390 test. These are research-dataset results, not app-store data.

**Theme discovery** (k = 4, 1752 sentences; majority-class purity 55%)

| Method | Purity | NMI |
|---|---|---|
| **MiniLM embeddings + k-means** | **78.9%** | 0.42 |
| TF-IDF + k-means | 62.9% | 0.19 |

Choosing k automatically by silhouette is weak (it picked k = 3 here), which is why the UI exposes a theme-count slider.

**Sentiment (positive vs negative)**

| Model | Accuracy | Latency |
|---|---|---|
| **Embeddings + logistic regression (in browser)** | **87.4%** (n=390, macro-F1 0.84) | ~1 ms |
| Majority class | 71.8% | – |
| openai/gpt-oss-120b (zero-shot) | 97.5% (n=80) | 1244 ms/review |
| openai/gpt-oss-20b (zero-shot) | 95.0% (n=80) | 1256 ms/review |
| qwen/qwen3.8-27b (zero-shot) | 98.8% (n=80) | 1495 ms/review |

Trade-off: zero-shot LLMs are more accurate but about 1,000x slower and cost tokens per review; the in-browser classifier is free, instant and private. LLM rows use an 80-review subsample, so compare with care.

**Model choice for writing:** on the same PRD prompt, `qwen/qwen3.8-27b` gave the best output (specific guardrails, real open questions, no invented numbers) at ~1.4 s, so it is the default.

Reproduce: `python3 scripts/fetch_data.py restaurants && npx tsx scripts/embed.ts && npx tsx scripts/eval_clusters.ts && python3 scripts/train_sentiment.py && GROQ_API_KEY=... npx tsx scripts/eval_llm.ts qwen/qwen3.8-27b openai/gpt-oss-120b`

## Run locally
`npm install && npm run dev` (set `GROQ_API_KEY` on Vercel, or use `vercel dev`, for theme naming and PRDs; the rest works without it).

## Stack
React, TypeScript, Vite, transformers.js (ONNX), scikit-learn (training only), Vercel serverless, Groq.
