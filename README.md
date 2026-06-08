# 🎬 Movie Genre Classifier

An NLP project that predicts a movie's **genre** from its plot/overview text. It uses a
custom **`FeatureUnion`** of **word-level** and **character-level TF-IDF n-grams** feeding a
linear classifier (**LinearSVC** or **Logistic Regression**), evaluated with **5-fold
stratified cross-validation** and served through a real-time **Gradio** interface.

> Dataset: ~53.9k movie plots across **27 genres** (IMDb plot data, Kaggle "Genre
> Classification Dataset"). The classes are heavily imbalanced (e.g. `drama` ≈ 13.5k vs
> `war` ≈ 132), which the model handles via balanced class weighting.

---

## Results

Full dataset (53,895 plots, all 27 genres), LinearSVC with balanced class weights,
80/20 stratified hold-out:

| Model     | Accuracy | Macro F1 | Weighted F1 |
|-----------|----------|----------|-------------|
| LinearSVC | **59.3%** | 0.386 | 0.580 |

The macro-F1 is dragged down by the rarest genres (e.g. `war` ≈ 132, `news` ≈ 181
samples), which is expected for a 27-class problem this imbalanced.

> Run `python model/train.py --model svm` (or `--model logreg`) for the full 5-fold
> stratified cross-validation, which prints accuracy, macro-F1 and weighted-F1.

---

## Project structure

```
movie-genre-classifier/
├── app.py                       # Gradio app (entry point for the demo UI)
├── app/
│   └── recommender.py           # (WIP) similarity-based recommender placeholder
├── data/
│   ├── prepare_movies.py        # cleans raw Kaggle TXT -> movies.csv (plot,genre)
│   ├── movies.csv               # cleaned training data (~53.9k rows)  [large]
│   ├── movies_full.csv          # full/debug export                    [large]
│   └── train_data.txt           # raw Kaggle dump                      [large]
├── model/
│   ├── train.py                 # FeatureUnion(word+char TF-IDF) + LinearSVC/LogReg, 5-fold CV
│   ├── predict.py               # MovieGenrePredictor inference wrapper
│   └── model.joblib             # trained artifact (NOT in git — produced by train.py)
├── tools/                       # one-off data-conversion helpers
├── requirements.txt
└── README.md
```

> **Note:** `model.joblib` is git-ignored (it is ~68 MB). You must **train once** before
> running the app — see below.

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

**1. (Optional) Rebuild the dataset** from the raw Kaggle TXT — already provided as
`data/movies.csv`, so you can skip this:

```bash
python data/prepare_movies.py
```

**2. Train the model** (produces `model/model.joblib`):

```bash
# LinearSVC (default) on the full dataset
python model/train.py --model svm

# Logistic Regression instead
python model/train.py --model logreg

# Faster smoke test on a stratified subsample
python model/train.py --model svm --sample 8000
```

Useful flags: `--sample N` (stratified subsample, `0` = full data), `--top_k K` (keep only
the K most frequent genres), `--balance` (undersample classes to equal size), `--svm_c` /
`--lr_c` (regularization strength).

**3. Launch the demo UI:**

```bash
python app.py
```

Then open the Gradio URL it prints, paste a movie plot, and get the top-K predicted genres
with probabilities.

---

## How it works

1. **Cleaning** (`data/prepare_movies.py`, `model/train.py`): parse the raw `ID ::: TITLE
   ::: GENRE ::: DESCRIPTION` records, normalize whitespace/case, drop very short plots.
2. **Features** (`model/train.py → make_features`): a `FeatureUnion` of
   - **word** TF-IDF, `ngram_range=(1, 3)`, English stop-words, `sublinear_tf`
   - **char** TF-IDF, `analyzer="char_wb"`, `ngram_range=(3, 5)`
   Character n-grams add robustness to spelling/morphology; word n-grams capture phrases.
3. **Model**: `LinearSVC` or multinomial `LogisticRegression`, both with
   `class_weight="balanced"` to counter the heavy genre imbalance.
4. **Evaluation**: `StratifiedKFold(n_splits=5)` reporting accuracy, macro-F1 and
   weighted-F1.
5. **Serving** (`app.py`): the fitted pipeline is wrapped by `MovieGenrePredictor`
   (`model/predict.py`) and exposed through a Gradio `Blocks` interface with top-K output.

## Future work

- Transformer embeddings (BERT/DistilBERT) for richer semantics
- Hyperparameter search (`RandomizedSearchCV` scaffolding already present in `train.py`)
- Finish the similarity-based recommender in `app/recommender.py`

## Contact

Sarfaraz Hussain — sarfaraz.hussain.work@gmail.com
