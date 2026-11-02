export function parseCsv(text: string): string[][] {
  const rows: string[][] = []; let row: string[] = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); cell = ''; if (row.some(x => x.trim())) rows.push(row); row = []; }
    else cell += c;
  }
  row.push(cell); if (row.some(x => x.trim())) rows.push(row);
  return rows;
}

export interface Parsed { headers: string[]; rows: string[][]; textCol: number; ratingCol: number }
export function detect(text: string): Parsed {
  const all = parseCsv(text); const headers = all[0]?.map(h => h.trim()) ?? []; const rows = all.slice(1);
  const find = (re: RegExp) => headers.findIndex(h => re.test(h.toLowerCase()));
  let textCol = find(/^(review|text|content|comment|body|feedback|message|review_text)$/);
  if (textCol < 0) { // fall back to the column with the longest average text
    let best = -1, len = 0; headers.forEach((_, j) => { const a = rows.slice(0, 50).reduce((s, r) => s + (r[j]?.length ?? 0), 0); if (a > len) { len = a; best = j; } }); textCol = best;
  }
  return { headers, rows, textCol, ratingCol: find(/^(rating|stars|score|star_rating)$/) };
}
export const toCsv = (rows: (string | number)[][]) => rows.map(r => r.map(v => { const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(',')).join('\n');
