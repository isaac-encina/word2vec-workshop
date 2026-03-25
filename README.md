# 🧠 Word2Vec Workshop — UWindsor AI Club

> **Build Word2Vec from scratch in PyTorch — understand the paper that changed NLP forever.**

Based on: [Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781)

![Tests](https://github.com/UWindsor-AI-Club/word2vec-workshop/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 What You'll Build

By the end of this workshop you'll have trained a model that understands word meaning from scratch. Type any word and it finds semantically similar ones:

```
most_similar("good")   →  great, excellent, wonderful, superb, fantastic
most_similar("film")   →  movie, picture, story, cinema, director
most_similar("scary")  →  terrifying, creepy, frightening, spooky, horror
```

All trained on your laptop. No GPU required. Runs in **under 5 minutes**.

---

## 📋 Prerequisites

| Tool | Download |
|------|----------|
| Python 3.10+ | https://python.org |
| Git | https://git-scm.com |

No prior AI knowledge required.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/UWindsor-AI-Club/word2vec-workshop.git
cd word2vec-workshop
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook
```

A browser tab opens. Click `word2vec_workshop.ipynb` and run each cell with **Shift + Enter**.

> 💡 First run downloads the IMDB dataset (~80 MB, cached after that).
> Total training time on a standard laptop: **3–5 minutes**.

---

## 🧪 Running the Tests

```bash
python test_word2vec.py
```

Expected output:

```
═══════════════════════════════════════════════════════
  Word2Vec Workshop — Test Suite
═══════════════════════════════════════════════════════
── Preprocessing ──────────────────────────────────────
✅ PASS  Lowercase applied
✅ PASS  HTML tags removed
✅ PASS  Punctuation removed
...
  Results: 35/35 passed  🎉
═══════════════════════════════════════════════════════
```

---

## 📂 Repo Structure

```
word2vec-workshop/
│
├── word2vec_workshop.ipynb   ← Main workshop notebook (10 steps)
├── test_word2vec.py          ← Automated test suite (35 tests)
├── requirements.txt          ← Python dependencies
├── setup.sh                  ← One-command setup for Mac/Linux
├── README.md                 ← This file
└── .github/
    └── workflows/
        └── tests.yml         ← GitHub Actions CI — runs on every push
```

---

## 📖 What Is Word2Vec? (Plain English)

### The problem
Computers don't understand words. Traditionally they treated every word as just an ID number — "cat" = 4271, "dog" = 892. There's no connection between them. The computer has no idea they're both animals.

### The solution
What if every word was a list of 100 numbers — a *vector* — and similar words had similar numbers? After training, "cat" and "dog" end up with nearby coordinates. "cat" and "skyscraper" end up far apart. Nobody programs these relationships in — the model figures them out just by reading text.

### How it learns: Skip-gram
The model reads *"the movie was absolutely brilliant"* and notices "brilliant" keeps appearing near "movie", "wonderful", "amazing". It nudges those words' coordinates closer together. Do this across millions of sentences and the coordinates get really smart.

### The magic: vector arithmetic
After training you can do maths with meaning:

```
King − Man + Woman ≈ Queen
Paris − France + Germany ≈ Berlin
walked − walk + run ≈ ran
```

Nobody told the model what gender, tense, or geography is. It learned the geometry of language by itself.

### Why this paper mattered
Before 2013, training good word vectors took weeks on expensive hardware. This paper got better results in under a day with two tricks:
1. **No hidden layer** — 10× faster than previous models
2. **Negative Sampling** — instead of checking all 1M vocab words per step, just check 5 random wrong ones

The concept of "represent things as vectors and measure similarity" is still the foundation of every modern language model including ChatGPT.

---

## 📓 Notebook Walkthrough

| Step | What happens |
|------|-------------|
| 1 | Install packages |
| 2 | Load 5,000 IMDB movie reviews from HuggingFace |
| 3 | Clean text — lowercase, strip HTML, tokenise |
| 4 | Build a vocabulary of the 10,000 most common words |
| 5 | Generate (target, context) Skip-gram pairs with window=2 |
| 6 | Build negative sampling noise distribution (freq^0.75 trick) |
| 7 | Define PyTorch Dataset — pre-samples negatives for speed |
| 8 | Define the model — two embedding tables, no hidden layer |
| 9 | Train with linear learning rate decay (as in the paper) |
| 10 | Query nearest neighbours with cosine similarity |
| Bonus | Vector arithmetic + PCA 2D visualisation + save/load |

---

## ⚙️ Workshop Knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NUM_REVIEWS` | 5,000 | More data = better vectors, slower training |
| `VOCAB_SIZE` | 10,000 | Larger = richer vocabulary |
| `WINDOW_SIZE` | 2 | Wider = captures longer-range relationships |
| `EMBED_DIM` | 100 | Higher = more expressive, slower |
| `NEG_SAMPLES` | 5 | More negatives = more stable training |
| `EPOCHS` | 3 | More passes = better vectors |

---

## 🔬 Things to Try After the Workshop

- Train on the full 25k IMDB dataset (`NUM_REVIEWS = 25000`)
- Switch to **CBOW** — predict center word from surrounding context
- Try vector arithmetic: `analogy("good", "great", "terrible")` → ?
- Swap IMDB for a different domain — news, tweets, books
- Increase `EMBED_DIM` to 300 and compare results

---

## 🤝 Contributing

This repo is maintained by the **UWindsor AI Club**.
Pull requests, issues, and suggestions are welcome!

- Repo: https://github.com/UWindsor-AI-Club/word2vec-workshop
- Issues: https://github.com/UWindsor-AI-Club/word2vec-workshop/issues

---

## 📄 License

MIT — use freely for teaching, learning, and building.

---

*Based on: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781*
