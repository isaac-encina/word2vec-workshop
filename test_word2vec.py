"""
test_word2vec.py
----------------
Automated tests for the Word2Vec workshop notebook.
Run with:  python test_word2vec.py

Tests every major component without needing Jupyter or the full dataset.
Uses a tiny synthetic corpus so tests finish in < 30 seconds.
"""

import sys
import re
import math
import random
import numpy as np
from collections import Counter
from itertools import chain

import torch
import torch.nn as nn

# ── reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"{status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, condition))
    return condition


# ════════════════════════════════════════════════════════════
# Minimal re-implementation of notebook functions (copy-pasted
# from the notebook so tests are self-contained)
# ════════════════════════════════════════════════════════════

# ── Preprocessing ────────────────────────────────────────────
def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

# ── Skip-gram pair generator ─────────────────────────────────
def generate_skipgram_pairs(encoded_sentences, window=2, unk_idx=0):
    pairs = []
    for sentence in encoded_sentences:
        n = len(sentence)
        for center_pos, center_word in enumerate(sentence):
            if center_word == unk_idx:
                continue
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                ctx_pos = center_pos + offset
                if 0 <= ctx_pos < n:
                    pairs.append((center_word, sentence[ctx_pos]))
    return pairs

# ── Model ────────────────────────────────────────────────────
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_emb  = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.target_emb.weight,  -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.context_emb.weight, -0.5/embed_dim, 0.5/embed_dim)

    def forward(self, targets, contexts, negatives):
        t_vecs = self.target_emb(targets)
        c_vecs = self.context_emb(contexts)
        n_vecs = self.context_emb(negatives)
        pos_score = torch.sum(t_vecs * c_vecs, dim=1)
        pos_loss  = torch.nn.functional.logsigmoid(pos_score)
        neg_scores = torch.bmm(n_vecs, t_vecs.unsqueeze(2)).squeeze(2)
        neg_loss   = torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

# ── most_similar ─────────────────────────────────────────────
def build_most_similar(embeddings, word2idx, idx2word):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    norm_emb = embeddings / norms

    def most_similar(word, k=5):
        word = word.lower()
        if word not in word2idx:
            return []
        idx = word2idx[word]
        scores = norm_emb @ norm_emb[idx]
        scores[idx] = -np.inf
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(idx2word[i], float(scores[i])) for i in top_k]

    return most_similar, norm_emb

# ════════════════════════════════════════════════════════════
# Synthetic corpus for fast testing
# ════════════════════════════════════════════════════════════
CORPUS = [
    "the movie was great and the acting was superb",
    "terrible film with awful acting and bad story",
    "a wonderful story with brilliant performances",
    "the worst movie i have ever seen it was horrible",
    "great film great story great acting loved it",
    "bad acting bad story bad film do not watch",
    "the director created a masterpiece of cinema",
    "outstanding performances from all the cast members",
    "completely boring and utterly terrible experience",
    "loved every moment of this incredible film",
] * 50   # repeat to get enough training signal


# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  Word2Vec Workshop — Test Suite")
print("=" * 55)
print()

# ── TEST 1: Preprocessing ────────────────────────────────────
print("── Preprocessing ──────────────────────────────────────")
t = preprocess("Hello WORLD! <br> This is a TEST... 123")
check("Lowercase applied",       all(c.islower() or c == " " for c in " ".join(t)))
check("HTML tags removed",       "br" not in t)
check("Punctuation removed",     "!" not in t and "..." not in t)
check("Numbers removed",         "123" not in t)
check("Returns a list",          isinstance(t, list))
check("Non-empty output",        len(t) > 0)
print()

# ── TEST 2: Vocabulary ───────────────────────────────────────
print("── Vocabulary ─────────────────────────────────────────")
tokenized = [preprocess(s) for s in CORPUS]
all_tokens = list(chain.from_iterable(tokenized))
freq = Counter(all_tokens)
VOCAB_SIZE = 50
vocab_words = [w for w, c in freq.most_common(VOCAB_SIZE) if c >= 2]
vocab_words = ["<UNK>"] + vocab_words
word2idx = {w: i for i, w in enumerate(vocab_words)}
idx2word = {i: w for w, i in word2idx.items()}
UNK_IDX = 0

check("Vocabulary created",          len(word2idx) > 1)
check("<UNK> at index 0",            word2idx["<UNK>"] == 0)
check("Common word in vocab",        "great" in word2idx)
check("idx2word is inverse",         idx2word[word2idx["great"]] == "great")
check("Vocab size capped correctly", len(word2idx) <= VOCAB_SIZE + 1)

encoded = [[word2idx.get(w, UNK_IDX) for w in s] for s in tokenized]
check("Encoding returns ints",       all(isinstance(i, int) for i in encoded[0]))
check("UNK used for unknowns",       UNK_IDX in [word2idx.get(w, UNK_IDX)
                                                  for w in ["xyzzy", "qqqq"]])
print()

# ── TEST 3: Skip-gram pairs ──────────────────────────────────
print("── Skip-gram Pairs ────────────────────────────────────")
pairs_w2 = generate_skipgram_pairs(encoded, window=2, unk_idx=UNK_IDX)
pairs_w1 = generate_skipgram_pairs(encoded, window=1, unk_idx=UNK_IDX)

check("Pairs generated",           len(pairs_w2) > 0)
check("Window=2 > window=1 pairs", len(pairs_w2) > len(pairs_w1))
check("Each pair is a 2-tuple",    all(len(p) == 2 for p in pairs_w2[:100]))
check("No UNK as center",          all(t != UNK_IDX for t, _ in pairs_w2))
check("Indices are valid ints",    all(isinstance(t, int) and isinstance(c, int)
                                       for t, c in pairs_w2[:100]))

# Verify pair logic on a known sentence
simple = [[1, 2, 3, 4, 5]]   # no UNK, window=1
sp = generate_skipgram_pairs(simple, window=1, unk_idx=0)
check("Center 2 → context 1 & 3", (2, 1) in sp and (2, 3) in sp)
check("Center 2 → NOT context 4", (2, 4) not in sp)
print()

# ── TEST 4: Negative sampling distribution ──────────────────
print("── Negative Sampling Distribution ─────────────────────")
counts = np.array([freq.get(idx2word.get(i, ""), 1)
                   for i in range(len(word2idx))], dtype=np.float32)
counts[UNK_IDX] = 0
noise_dist = counts ** 0.75
noise_dist /= noise_dist.sum()

check("Distribution sums to 1",   abs(noise_dist.sum() - 1.0) < 1e-5)
check("UNK never sampled",        noise_dist[UNK_IDX] == 0.0)
check("All probabilities >= 0",   (noise_dist >= 0).all())
check("Most frequent has highest prob",
      noise_dist[word2idx["the"]] > noise_dist[word2idx.get("masterpiece", 1)])

# Spot-check freq^0.75 smoothing: rare word should have relatively
# higher prob than raw frequency would give
raw_dist = counts / counts.sum()
# Find a word that's much rarer than "the"
rare_candidates = [w for w, c in freq.most_common()[-10:] if w in word2idx]
if rare_candidates:
    rw = rare_candidates[0]
    ri = word2idx[rw]
    ti = word2idx["the"]
    ratio_noise = noise_dist[ri] / (noise_dist[ti] + 1e-12)
    ratio_raw   = raw_dist[ri]   / (raw_dist[ti]   + 1e-12)
    check("Freq^0.75 boosts rare words vs raw freq", ratio_noise > ratio_raw)
print()

# ── TEST 5: Model architecture ───────────────────────────────
print("── Model Architecture ─────────────────────────────────")
VOCAB  = len(word2idx)
DIM    = 16   # tiny for fast tests
model  = SkipGramNegSampling(VOCAB, DIM)

check("Model has target_emb",        hasattr(model, "target_emb"))
check("Model has context_emb",       hasattr(model, "context_emb"))
check("Embedding shape correct",
      tuple(model.target_emb.weight.shape) == (VOCAB, DIM))
check("Two separate embedding tables",
      model.target_emb.weight.data_ptr() != model.context_emb.weight.data_ptr())
print()

# ── TEST 6: Forward pass & loss ──────────────────────────────
print("── Forward Pass & Loss ────────────────────────────────")
B, K = 8, 5
targets   = torch.randint(1, VOCAB, (B,))
contexts  = torch.randint(1, VOCAB, (B,))
negatives = torch.randint(1, VOCAB, (B, K))

try:
    loss = model(targets, contexts, negatives)
    check("Forward pass runs without error", True)
    check("Loss is a scalar",                loss.shape == torch.Size([]))
    check("Loss is finite",                  torch.isfinite(loss).item())
    check("Loss is positive",                loss.item() > 0)
    check("Loss is reasonable (< 20)",       loss.item() < 20)
except Exception as e:
    check("Forward pass runs without error", False, str(e))
    check("Loss is a scalar",                False)
    check("Loss is finite",                  False)
    check("Loss is positive",                False)
    check("Loss is reasonable (< 20)",       False)
print()

# ── TEST 7: Backward pass / gradient flow ───────────────────
print("── Gradients ──────────────────────────────────────────")
model.zero_grad()
loss = model(targets, contexts, negatives)
loss.backward()

t_grad = model.target_emb.weight.grad
c_grad = model.context_emb.weight.grad
check("Target embedding gets gradients",  t_grad is not None)
check("Context embedding gets gradients", c_grad is not None)
check("Gradients are finite",
      torch.isfinite(t_grad).all().item() and torch.isfinite(c_grad).all().item())
print()

# ── TEST 8: Mini training loop (loss decreases) ─────────────
print("── Mini Training Loop ─────────────────────────────────")
model2    = SkipGramNegSampling(VOCAB, DIM)
optimizer = torch.optim.SGD(model2.parameters(), lr=0.025)

losses = []
for _ in range(60):
    t_ = torch.randint(1, VOCAB, (32,))
    c_ = torch.randint(1, VOCAB, (32,))
    n_ = torch.randint(1, VOCAB, (32, 5))
    optimizer.zero_grad()
    l = model2(t_, c_, n_)
    l.backward()
    optimizer.step()
    losses.append(l.item())

first_avg = sum(losses[:10]) / 10
last_avg  = sum(losses[-10:]) / 10
check("Loss decreases over training",    last_avg < first_avg,
      f"first={first_avg:.3f} last={last_avg:.3f}")
check("Loss doesn't explode",            all(math.isfinite(l) for l in losses))
print()

# ── TEST 9: most_similar ─────────────────────────────────────
print("── most_similar() ─────────────────────────────────────")

# Train a slightly bigger model for a few epochs so similarity is meaningful
FULL_DIM = 32
full_model = SkipGramNegSampling(VOCAB, FULL_DIM)
opt = torch.optim.SGD(full_model.parameters(), lr=0.05)

neg_dist = noise_dist.copy()
all_pair_list = pairs_w2

for _ in range(5):   # 5 quick epochs on synthetic data
    random.shuffle(all_pair_list)
    for i in range(0, min(len(all_pair_list), 2000), 64):
        batch = all_pair_list[i:i+64]
        if len(batch) < 2:
            continue
        bt = torch.tensor([p[0] for p in batch], dtype=torch.long)
        bc = torch.tensor([p[1] for p in batch], dtype=torch.long)
        bn = torch.tensor(
            np.random.choice(VOCAB, size=(len(batch), 5), p=neg_dist),
            dtype=torch.long)
        opt.zero_grad()
        full_model(bt, bc, bn).backward()
        opt.step()

with torch.no_grad():
    emb = full_model.target_emb.weight.cpu().numpy()

most_similar, norm_emb = build_most_similar(emb, word2idx, idx2word)

res = most_similar("great", k=5)
check("Returns a list",                isinstance(res, list))
check("Returns k results",             len(res) == 5)
check("Each item is (str, float)",
      all(isinstance(w, str) and isinstance(s, float) for w, s in res))
check("Scores are in [-1, 1]",         all(-1.01 <= s <= 1.01 for _, s in res))
check("Scores are descending",
      all(res[i][1] >= res[i+1][1] for i in range(len(res)-1)))
check("Query word not in results",     all(w != "great" for w, _ in res))

# OOV word
oov = most_similar("zzzznotaword", k=5)
check("OOV word returns empty list",   oov == [])
print()

# ── TEST 10: cosine similarity sanity ────────────────────────
print("── Cosine Similarity Sanity ───────────────────────────")
# Two identical vectors should have cosine sim = 1
v = np.random.rand(FULL_DIM).astype(np.float32)
v_norm = v / np.linalg.norm(v)
check("Identical vectors → sim ≈ 1",
      abs(float(v_norm @ v_norm) - 1.0) < 1e-5)

# Orthogonal vectors should have cosine sim = 0
a = np.zeros(FULL_DIM, dtype=np.float32); a[0] = 1.0
b = np.zeros(FULL_DIM, dtype=np.float32); b[1] = 1.0
check("Orthogonal vectors → sim ≈ 0",
      abs(float(a @ b)) < 1e-5)
print()

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
print("=" * 55)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"   ({failed} FAILED)")
    print()
    print("Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
else:
    print("  🎉")
print("=" * 55)

sys.exit(0 if failed == 0 else 1)
