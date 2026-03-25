# 🧠 Word2Vec Workshop — UWindsor AI Club

> **Build Word2Vec from scratch in PyTorch — step-by-step in an interactive notebook.**

Based on: *Efficient Estimation of Word Representations in Vector Space* (Mikolov et al., 2013)

---

## 🎯 What You'll Build

In this workshop, you will implement **Word2Vec (Skip-gram with Negative Sampling)** from scratch and train it on real data.

Your model will:

* Load IMDB movie reviews
* Preprocess text (lowercase + tokenization)
* Build a vocabulary (~10,000 words)
* Generate (target, context) training pairs
* Train a Skip-gram model with ~5 negative samples
* Learn word embeddings in **under 5 minutes on CPU**
* Query similar words using cosine similarity

Example:

```
most_similar("good")
→ great, excellent, amazing, fantastic, wonderful
```

---

## 📋 Prerequisites

| Tool   | Required                    |
| ------ | --------------------------- |
| Python | 3.10 – 3.12 (recommended)   |
| pip    | Latest version              |
| Git    | Optional (for cloning repo) |

⚠️ **Important:**
Python 3.14 (Microsoft Store version) may cause issues with Jupyter.
If you run into problems, install Python from https://python.org instead.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/UWindsor-AI-Club/word2vec-workshop.git
cd word2vec-workshop
```

---

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 3. Launch Jupyter (IMPORTANT FIX)

If `jupyter` command does NOT work (common on Windows):

```bash
python -m jupyter notebook
```

✅ This avoids PATH issues.

---

### 4. Open the notebook

* A browser will open automatically
* Click:

```
word2vec_workshop.ipynb
```

---

### 5. Run the notebook

* Click the first cell
* Press:

```
Shift + Enter
```

* Repeat for each cell (top → bottom)

---

## ⚠️ Common Issues (READ THIS)

### ❌ "jupyter is not recognized"

Fix:

```bash
python -m jupyter notebook
```

---

### ❌ Notebook missing

Make sure this file exists:

```
word2vec_workshop.ipynb
```

If not:

```bash
git pull
```

---

### ❌ Nothing happens

You must **run cells manually**:

* Click a cell
* Press **Shift + Enter**

---

### ❌ Slow or failing training

Reduce dataset size in notebook:

```python
NUM_REVIEWS = 2000
```

---

## 🧠 What the Notebook Actually Does

This is an **interactive pipeline** — not a single script.

| Step | What happens                                |
| ---- | ------------------------------------------- |
| 1    | Load IMDB dataset (subset for speed)        |
| 2    | Clean text (lowercase + tokenize)           |
| 3    | Build vocabulary (~10k most frequent words) |
| 4    | Convert words → indices                     |
| 5    | Generate (target, context) pairs            |
| 6    | Sample negative examples (~5 per pair)      |
| 7    | Define Skip-gram model (PyTorch)            |
| 8    | Train embeddings                            |
| 9    | Compute cosine similarity                   |
| 10   | Query similar words                         |

---

## 📓 How to Use the Notebook (IMPORTANT)

This is NOT a normal Python script.

You do NOT run:

```bash
python word2vec_workshop.ipynb
```

Instead:

👉 Run **one cell at a time**

Each cell:

* Executes code
* Shows output directly below

---

### Example

Cell:

```python
print(tokens[:10])
```

Output:

```
['this', 'movie', 'was', 'really', 'good']
```

---

## 🔍 Understanding Outputs

### During preprocessing

```
['this', 'movie', 'was', 'great']
```

→ text successfully tokenized

---

### During training

```
Epoch 1: Loss = 2.3
Epoch 2: Loss = 1.8
Epoch 3: Loss = 1.5
```

→ model is learning (loss decreasing)

---

### After training

```
most_similar("film")
→ movie, cinema, story
```

→ embeddings are meaningful

---

## ⚙️ Key Parameters

| Parameter     | Default | Effect           |
| ------------- | ------- | ---------------- |
| `NUM_REVIEWS` | 5000    | Training size    |
| `VOCAB_SIZE`  | 10000   | Vocabulary size  |
| `WINDOW_SIZE` | 2       | Context window   |
| `EMBED_DIM`   | 100     | Embedding size   |
| `NEG_SAMPLES` | 5       | Negative samples |
| `EPOCHS`      | 3       | Training passes  |

---

## 🧪 Running Tests

```bash
python test_word2vec.py
```

Expected:

```
Results: 35/35 passed 🎉
```

---

## 📂 Repo Structure

```
word2vec-workshop/
│
├── word2vec_workshop.ipynb   ← Main notebook (REQUIRED)
├── test_word2vec.py          ← Tests
├── requirements.txt          ← Dependencies
├── setup.sh                  ← Auto setup script
└── README.md                 ← This file
```

---

## 🔬 What You’re Learning

* How Word2Vec works internally
* How embeddings capture meaning
* How negative sampling speeds up training
* How modern NLP models represent language

---

## 🚀 After the Workshop

Try:

* Increase `EMBED_DIM` to 300
* Use full dataset (`NUM_REVIEWS = 25000`)
* Implement CBOW
* Try word analogies

---

## 🤝 Contributing

Maintained by **UWindsor AI Club**

---

## 📄 License

MIT

---

## 🧠 Final Insight

> You are building a **map of language** where similar words live close together in space.

---
