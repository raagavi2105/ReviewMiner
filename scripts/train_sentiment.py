"""Train a logistic-regression sentiment head on MiniLM embeddings; export weights for in-browser inference."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
tr, te = json.load(open("data/train.json")), json.load(open("data/test.json"))
Xtr, Xte = np.array(tr["emb"]), np.array(te["emb"])
ytr = np.array([r["polarity"] == "positive" for r in tr["rows"]], int)
yte = np.array([r["polarity"] == "positive" for r in te["rows"]], int)
clf = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, ytr)
p = clf.predict(Xte)
res = {"n_train": len(ytr), "n_test": len(yte), "majority_baseline_acc": float(max(yte.mean(), 1 - yte.mean())),
       "lr_accuracy": float(accuracy_score(yte, p)), "lr_macro_f1": float(f1_score(yte, p, average="macro"))}
json.dump(res, open("eval-sentiment-lr.json", "w"), indent=2); print(res)
json.dump({"w": [round(float(x), 5) for x in clf.coef_[0]], "b": float(clf.intercept_[0])}, open("public/sentiment-lr.json", "w"))
