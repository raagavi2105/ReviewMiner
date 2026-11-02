"""Download SemEval-2014 Task 4 (ABSA) sentences from Hugging Face. Human-labeled aspect categories + polarity."""
import json, urllib.request, urllib.parse
def rows(cfg, split):
    out, off = [], 0
    while True:
        u = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({"dataset":"alexcadillon/SemEval2014Task4","config":cfg,"split":split,"offset":off,"length":100})
        d = json.load(urllib.request.urlopen(u))
        r = d["rows"]
        if not r: break
        out += [x["row"] for x in r]; off += 100
        if off >= d["num_rows_total"]: break
    return out
import sys
for cfg in (sys.argv[1:] or ["restaurants","laptops"]):
    for sp in ["train","test"]:
        r = rows(cfg, sp); json.dump(r, open(f"data/{cfg}_{sp}.json","w")); print(cfg, sp, len(r))
