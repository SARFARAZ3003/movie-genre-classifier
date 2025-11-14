# model/train.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# --- Paths ---
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


# --------- Args ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV with columns: plot,genre")
    p.add_argument("--sample", type=int, default=0,
                   help="approx rows to use (0 = use FULL data)")
    p.add_argument("--min_words", type=int, default=20,
                   help="drop rows where plot word-count < this")
    p.add_argument("--top_k", type=int, default=0,
                   help="keep only top-K most frequent genres (0 = keep all)")
    p.add_argument("--balance", action="store_true",
                   help="undersample all classes to the same size after filters")
    p.add_argument("--per_class_cap", type=int, default=0,
                   help="max samples per class when --balance is used (0 = use min class size)")

    # model choices
    p.add_argument("--model", choices=["svm", "logreg"], default="svm")
    p.add_argument("--svm_c", type=float, default=2.0)
    p.add_argument("--lr_c", type=float, default=6.0)
    return p.parse_args()


# --------- Data ----------
def load_data(path, min_words):
    df = pd.read_csv(path)
    df["plot"] = df["plot"].fillna("").astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip().str.lower()
    if min_words > 0:
        df = df[df["plot"].str.split().str.len() >= min_words].reset_index(drop=True)
    return df


# --------- Features ----------
def make_features():
    word_v = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 3), sublinear_tf=True,
        max_df=0.95, min_df=2, dtype=np.float32
    )
    char_v = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        sublinear_tf=True, min_df=2, dtype=np.float32
    )
    return FeatureUnion([("word", word_v), ("char", char_v)])


def build_pipeline(model_name: str, C_svm: float, C_lr: float):
    feats = make_features()
    if model_name == "svm":
        clf = LinearSVC(C=C_svm, class_weight="balanced", random_state=42)
    else:
        # multinomial works very well on TF-IDF for long text
        clf = LogisticRegression(
            max_iter=4000, solver="saga", multi_class="multinomial",
            C=C_lr, class_weight="balanced", n_jobs=1, random_state=42
        )
    return Pipeline(steps=[("tfidf", feats), ("clf", clf)])


def stratified_sample(df: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
    """Sample approximately target_n rows while keeping per-genre proportions and
    never taking fewer than 50 per class (when available)."""
    if target_n <= 0 or target_n >= len(df):
        return df
    n_classes = df["genre"].nunique()
    per_class = max(50, target_n // n_classes)
    rng = np.random.default_rng(seed)
    sampled = (
        df.groupby("genre", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed))
          .reset_index(drop=True)
    )
    # If still under target because some classes are tiny, top up randomly
    if len(sampled) < target_n:
        need = target_n - len(sampled)
        rest = df.drop(sampled.index, errors="ignore")
        if len(rest) > 0:
            take = min(need, len(rest))
            extra_idx = rng.choice(rest.index.to_numpy(), size=take, replace=False)
            sampled = pd.concat([sampled, rest.loc[extra_idx]], ignore_index=True)
    return sampled


def main():
    args = parse_args()
    data_path = Path(args.data)

    print(f"📂 Loading data from: {data_path}")
    df = load_data(data_path, args.min_words)
    print(f"Total rows after cleaning: {len(df)}")

    # Optional: keep top-K frequent genres
    if args.top_k and args.top_k > 0:
        keep = df["genre"].value_counts().nlargest(args.top_k).index
        df = df[df["genre"].isin(keep)].reset_index(drop=True)
        print(f"Kept top-{args.top_k} genres. Rows now: {len(df)}")

    # Optional: stratified sample
    if args.sample and args.sample > 0:
        df = stratified_sample(df, args.sample, seed=42)
        print(f"Using {len(df)} rows after stratified sample.")
    else:
        print("Using FULL dataset (no sampling).")

    # Optional: balance by undersampling
    if args.balance:
        counts = df["genre"].value_counts()
        target = counts.min() if args.per_class_cap == 0 else min(args.per_class_cap, counts.min())
        df = (
            df.groupby("genre", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), target), random_state=42))
              .reset_index(drop=True)
        )
        print(f"Balanced ~{target} per class. Rows now: {len(df)}")

    classes = sorted(df["genre"].unique())
    print(f"Genres: {classes}")

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df["genre"])
    X = df["plot"]

    pipe = build_pipeline(args.model, args.svm_c, args.lr_c)

    # CV
    print("\n=== 🧠 5-fold CV ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
    f1_macro = cross_val_score(pipe, X, y, cv=skf,
                               scoring=make_scorer(f1_score, average="macro"), n_jobs=1)
    f1_weighted = cross_val_score(pipe, X, y, cv=skf,
                                  scoring=make_scorer(f1_score, average="weighted"), n_jobs=1)

    print(f"Accuracy:     mean={acc_scores.mean():.3f} (± {acc_scores.std():.3f})")
    print(f"Macro F1:     mean={f1_macro.mean():.3f} (± {f1_macro.std():.3f})")
    print(f"Weighted F1:  mean={f1_weighted.mean():.3f} (± {f1_weighted.std():.3f})")

    print("\n⏳ Training final model on ALL selected data...")
    pipe.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labeler": le, "classes": classes}, MODEL_PATH)
    print(f"🎉 Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()





































# # model/train.py

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# import argparse

# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import make_scorer, f1_score
# from sklearn.preprocessing import LabelEncoder

# # --- Paths ---
# DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
# MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
#     p.add_argument("--sample", type=int, default=5000, help="sample size")
#     return p.parse_args()

# # --- Data ---
# def load_data(path):
#     df = pd.read_csv(path)
#     df["plot"] = df["plot"].fillna("").astype(str).str.strip()
#     df["genre"] = df["genre"].astype(str).str.strip()
#     return df

# # --- Pipeline ---
# def build_pipeline():
#     return Pipeline(steps=[
#         ("tfidf", TfidfVectorizer(
#             lowercase=True,
#             stop_words="english",
#             ngram_range=(1, 2),
#             sublinear_tf=True,
#             max_df=0.95,
#             min_df=3,
#             max_features=30000,
#             dtype=np.float32
#         )),
#         ("clf", LogisticRegression(
#             max_iter=2000,
#             solver="liblinear",
#             multi_class="ovr",
#             C=2.0,
#             class_weight="balanced",
#             n_jobs=1,
#             random_state=42
#         ))
#     ])

# def main():
#     args = parse_args()
#     data_path = Path(args.data)

#     print(f"📂 Loading data from: {data_path}")
#     df = load_data(data_path)
#     print(f"Total rows: {len(df)}")

#     # ✅ Stratified Sample
#     SAMPLE = args.sample
#     print(f"🎯 Sampling approx: {SAMPLE} rows...")
#     df = df.groupby("genre", group_keys=False).apply(
#         lambda x: x.sample(min(len(x), max(50, SAMPLE // df['genre'].nunique())), random_state=42)
#     ).reset_index(drop=True)
#     print(f"✅ Using {len(df)} rows after stratified sample.")

#     classes = sorted(df["genre"].unique())
#     print(f"Genres: {classes}")

#     # Labels
#     le = LabelEncoder()
#     y = le.fit_transform(df["genre"])
#     X = df["plot"]

#     pipe = build_pipeline()

#     # ✅ CV - safe mode
#     print("\n=== 🧠 5-fold CV (memory safe mode) ===")
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
#     f1_macro = cross_val_score(pipe, X, y, cv=skf,
#                                scoring=make_scorer(f1_score, average="macro"),
#                                n_jobs=1)

#     print(f"✅ Accuracy: mean={acc_scores.mean():.3f} (± {acc_scores.std():.3f})")
#     print(f"✅ Macro F1: mean={f1_macro.mean():.3f} (± {f1_macro.std():.3f})")

#     print("\n⏳ Training final model...")
#     pipe.fit(X, y)

#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump({"pipeline": pipe, "labeler": le, "classes": classes}, MODEL_PATH)
#     print(f"🎉 Model saved to: {MODEL_PATH}")

# if __name__ == "__main__":
#     main()

















# # train my python ..... code

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# import argparse

# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import make_scorer, f1_score
# from sklearn.preprocessing import LabelEncoder

# # --- Paths ---
# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
# MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", default=str(DATA_PATH))
#     p.add_argument("--sample", default=5000, type=int, help="sample size")
#     return p.parse_args()

# # --- Data ---
# def load_data(path):
#     df = pd.read_csv(path)
#     df["plot"] = df["plot"].fillna("").astype(str).str.strip()
#     df["genre"] = df["genre"].astype(str).str.strip()
#     return df

# # --- Pipeline ---
# def build_pipeline():
#     return Pipeline(steps=[
#         ("tfidf", TfidfVectorizer(
#             lowercase=True,
#             stop_words="english",
#             ngram_range=(1, 2),
#             sublinear_tf=True,
#             max_df=0.95,
#             min_df=3,
#             max_features=30000,
#             dtype=np.float32
#         )),
#         ("clf", LogisticRegression(
#             # max_iter=2000,
#             # solver="liblinear",
#             # multi_class="ovr",
#             # C=2.0,
#             # class_weight="balanced",
#             # n_jobs=1,
#             # random_state=42
#         ))
#     ])

# def main():
#     args = parse_args()
#     data_path = Path(args.data)

#     df = load_data(data_path)

#     SAMPLE = args.sample
#     df = df.groupby("genre", group_keys=False).apply(
#         lambda x: x.sample(min(len(x), max(50, SAMPLE // df['genre'].nunique())), random_state=42)
#     ).reset_index(drop=True)

#     classes = sorted(df["genre"].unique())

#     le = LabelEncoder()
#     y = le.fit_transform(df["genre"])
#     X = df["plot"]

#     pipe = build_pipeline()

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
#     f1_macro = cross_val_score(
#         pipe, X, y, cv=skf,
#         scoring=make_scorer(f1_score, average="macro"),
#         n_jobs=1
#     )

#     pipe.fit(X, y)

#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump({"pipeline": pipe, "labeler": le, "classes": classes}, MODEL_PATH)

# if __name__ == "__main__":
#     main()


# model/train.py

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# from pathlib import Path
# import argparse
# import joblib
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import make_scorer, f1_score

# # --- Paths ---
# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
# MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


# # --------- Args ----------
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", default=str(DATA_PATH), help="path to CSV with columns: plot,genre")
#     p.add_argument("--sample", default=0, type=int,
#                    help="approx rows to use (0 or >=len(df) = use FULL data)")
#     p.add_argument("--min_words", default=10, type=int,
#                    help="drop rows where plot word-count < min_words")
#     p.add_argument("--min_class_count", default=0, type=int,
#                    help="drop classes with < this many samples (0 = keep all)")
#     p.add_argument("--svm_c", default=1.0, type=float, help="LinearSVC C")
#     return p.parse_args()


# # --------- Data ----------
# def load_data(path, min_words, min_class_count):
#     df = pd.read_csv(path)
#     # basic cleaning
#     df["plot"] = df["plot"].fillna("").astype(str).str.strip()
#     df["genre"] = df["genre"].astype(str).str.strip().str.lower()

#     # drop very short plots (noise)
#     if min_words > 0:
#         df = df[df["plot"].str.split().str.len() >= min_words].reset_index(drop=True)

#     # optionally drop ultra-rare classes
#     if min_class_count > 0:
#         counts = df["genre"].value_counts()
#         keep = counts[counts >= min_class_count].index
#         df = df[df["genre"].isin(keep)].reset_index(drop=True)

#     return df


# # --------- Model ----------
# def build_pipeline(C=1.0):
#     word_v = TfidfVectorizer(
#         lowercase=True,
#         stop_words="english",
#         ngram_range=(1, 3),
#         sublinear_tf=True,
#         max_df=0.9,
#         min_df=2,
#         max_features=None,   # can cap e.g. 200_000 if RAM tight
#         dtype=np.float32,
#     )
#     char_v = TfidfVectorizer(
#         analyzer="char_wb",
#         ngram_range=(3, 5),
#         sublinear_tf=True,
#         min_df=2,
#         max_features=None,   # can cap e.g. 100_000 if RAM tight
#         dtype=np.float32,
#     )

#     feats = FeatureUnion([("word", word_v), ("char", char_v)])

#     clf = LinearSVC(C=C, class_weight="balanced", random_state=42)

#     return Pipeline(steps=[("tfidf", feats), ("clf", clf)])


# def main():
#     args = parse_args()
#     data_path = Path(args.data)

#     print(f"Loading data from : {data_path}")
#     df = load_data(data_path, args.min_words, args.min_class_count)
#     total = len(df)
#     print(f"Total rows after cleaning: {total}")

#     # --- Stratified sampling (fixed) ---
#     SAMPLE = int(args.sample)
#     if SAMPLE <= 0 or SAMPLE >= total:
#         print("Using FULL dataset (no sampling).")
#     else:
#         frac = SAMPLE / total
#         print(f"Sampling approx: {SAMPLE} rows (~{frac:.2%}) ...")
#         df = (
#             df.groupby("genre", group_keys=False)
#             .apply(lambda g: g.sample(max(1, int(len(g) * frac)), random_state=42))
#             .reset_index(drop=True)
#         )
#     print(f"Using {len(df)} rows after stratified sample.")

#     classes = sorted(df["genre"].unique())
#     print(f"Genres: {classes}")

#     # Labels
#     le = LabelEncoder()
#     y = le.fit_transform(df["genre"])
#     X = df["plot"]

#     pipe = build_pipeline(C=args.svm_c)

#     # --- Cross-val ---
#     print("\n=== 5-fold CV (memory safe mode) ===")
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
#     f1_macro = cross_val_score(
#         pipe, X, y, cv=skf, scoring=make_scorer(f1_score, average="macro"), n_jobs=1
#     )
#     f1_weighted = cross_val_score(
#         pipe, X, y, cv=skf, scoring=make_scorer(f1_score, average="weighted"), n_jobs=1
#     )

#     print("\n=== 5-fold CV ===")
#     print(f"Accuracy:     mean={acc_scores.mean():.3f}  (± {acc_scores.std():.3f})")
#     print(f"Macro F1:     mean={f1_macro.mean():.3f}    (± {f1_macro.std():.3f})")
#     print(f"Weighted F1:  mean={f1_weighted.mean():.3f} (± {f1_weighted.std():.3f})")

#     # --- Final fit on ALL selected data ---
#     pipe.fit(X, y)

#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump({"pipeline": pipe, "labeler": le, "classes": classes}, MODEL_PATH)
#     print(f"\nSaved model to: {MODEL_PATH}")


# if __name__ == "__main__":
#     main()






















# # model/train.py

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# from pathlib import Path
# import argparse
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import html
# import unicodedata

# from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.ensemble import VotingClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import make_scorer, f1_score

# # xgboost (pip install xgboost)
# try:
#     from xgboost import XGBClassifier
#     HAS_XGB = True
# except Exception:
#     HAS_XGB = False

# # --- Paths ---
# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
# MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


# # ========= Text cleaning helpers =========
# _re_url = re.compile(r"https?://\S+|www\.\S+", re.I)
# _re_email = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
# _re_user = re.compile(r"[@#]\w+")
# _re_html = re.compile(r"<[^>]+>")
# _re_brackets = re.compile(r"\[(.*?)\]|\((.*?)\)")
# _re_multispace = re.compile(r"\s+")
# _re_ellipsis = re.compile(r"\.{2,}")
# _re_nonword_edges = re.compile(r"[^\w\s\-]")  # keep hyphen for 'sci-fi'
# _re_years = re.compile(r"\b(19|20)\d{2}\b")

# CONTRACTIONS = {
#     "can't":"cannot","won't":"will not","n't":" not","i'm":"i am","it's":"it is","that's":"that is",
#     "what's":"what is","there's":"there is","I've":"I have","i've":"i have","we're":"we are",
#     "they're":"they are","you're":"you are","isn't":"is not","aren't":"are not","wasn't":"was not",
#     "weren't":"were not","doesn't":"does not","don't":"do not","didn't":"did not","hasn't":"has not",
#     "haven't":"have not","hadn't":"had not","couldn't":"could not","shouldn't":"should not",
#     "wouldn't":"would not","I'll":"I will","i'll":"i will","we'll":"we will","they'll":"they will",
#     "he's":"he is","she's":"she is","who's":"who is","let's":"let us",
# }

# LABEL_ALIASES = {
#     "science fiction":"sci-fi",
#     "sci fi":"sci-fi",
#     "film noir":"film-noir",
#     "game show":"game-show",
#     "talk show":"talk-show",
#     "reality tv":"reality-tv",
#     "tv movie":"tv-movie",
# }

# def normalize_label(g: str) -> str:
#     s = g.strip().lower()
#     s = s.replace("_","-").replace("—","-").replace("–","-")
#     s = LABEL_ALIASES.get(s, s)
#     return s

# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         text = "" if text is None else str(text)

#     # html unescape & strip accents
#     t = html.unescape(text)
#     t = unicodedata.normalize("NFKD", t)
#     t = "".join(ch for ch in t if not unicodedata.combining(ch))

#     t = t.lower().strip()
#     t = _re_html.sub(" ", t)
#     t = _re_url.sub(" ", t)
#     t = _re_email.sub(" ", t)
#     t = _re_user.sub(" ", t)
#     t = _re_brackets.sub(" ", t)         # drop bracketed meta
#     t = _re_years.sub(" ", t)            # years rarely help generalization
#     # expand common contractions
#     for k,v in CONTRACTIONS.items():
#         t = t.replace(k, v)

#     t = _re_ellipsis.sub(" ", t)
#     # remove punctuation except hyphen (keep sci-fi)
#     t = _re_nonword_edges.sub(" ", t)
#     # collapse multiple dashes to single (e.g., multi-hyphen noise)
#     t = re.sub(r"-{2,}", "-", t)
#     # collapse spaces
#     t = _re_multispace.sub(" ", t).strip()
#     return t


# # --------- Args ----------
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", default=str(DATA_PATH), help="CSV with columns: plot,genre")
#     p.add_argument("--sample", default=0, type=int, help="0/full = use all rows")
#     p.add_argument("--min_words", default=12, type=int, help="drop rows where plot words < this")
#     # keep but default is to NOT use any label filtering/balancing
#     p.add_argument("--min_class_count", default=0, type=int)
#     p.add_argument("--top_k", type=int, default=0)
#     p.add_argument("--balance", action="store_true")
#     p.add_argument("--per_class_cap", type=int, default=0)

#     # model
#     p.add_argument("--model", choices=["logreg", "svm", "xgb", "vote"], default="logreg")
#     p.add_argument("--optimize_for", choices=["accuracy", "balanced"], default="accuracy")
#     p.add_argument("--lr_c", type=float, default=6.0)
#     p.add_argument("--svm_c", type=float, default=2.0)
#     p.add_argument("--calibrate_svm", action="store_true",
#                    help="wrap LinearSVC with CalibratedClassifierCV (gives predict_proba; slower)")
#     # tuning
#     p.add_argument("--tune", choices=["none", "light", "pro"], default="light")
#     p.add_argument("--tune_iter", type=int, default=20, help="n_iter for RandomizedSearch (light: 20, pro: 60)")
#     p.add_argument("--random_state", type=int, default=42)
#     return p.parse_args()


# # --------- Data ----------
# def load_data(path, min_words, min_class_count):
#     df = pd.read_csv(path)

#     # label normalize (fix hyphen/space variants)
#     df["genre"] = df["genre"].astype(str).map(normalize_label)

#     # text clean
#     df["plot"] = df["plot"].map(clean_text)

#     # drop duplicates & empties
#     before = len(df)
#     df = df[~df["plot"].isna() & (df["plot"].str.len() > 0)]
#     df = df.drop_duplicates(subset=["plot"]).reset_index(drop=True)

#     # drop very short plots (noise)
#     if min_words > 0:
#         df = df[df["plot"].str.split().str.len() >= min_words].reset_index(drop=True)

#     # optionally drop ultra-rare classes (keep 0 for all-genres accuracy runs)
#     if min_class_count > 0:
#         counts = df["genre"].value_counts()
#         keep = counts[counts >= min_class_count].index
#         df = df[df["genre"].isin(keep)].reset_index(drop=True)

#     print(f"Removed {before - len(df)} rows via cleaning/duplicates/short text.")
#     return df


# # --------- Features ----------
# def make_features():
#     # Slightly stricter min_df to cut spelling noise after cleaning
#     word_v = TfidfVectorizer(
#         lowercase=True, stop_words="english",
#         ngram_range=(1, 3), sublinear_tf=True,
#         strip_accents="unicode", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
#         max_df=0.98, min_df=3, max_features=None, dtype=np.float32
#     )
#     char_v = TfidfVectorizer(
#         analyzer="char_wb", ngram_range=(3, 6),
#         sublinear_tf=True, min_df=3, max_features=None, dtype=np.float32
#     )
#     return FeatureUnion([("word", word_v), ("char", char_v)])


# # --------- Model builders ----------
# def make_clf(args):
#     class_wt = None if args.optimize_for == "accuracy" else "balanced"

#     if args.model == "logreg":
#         return LogisticRegression(
#             max_iter=4000, solver="saga", multi_class="multinomial",
#             penalty="l2", C=args.lr_c, class_weight=class_wt, random_state=args.random_state
#         )

#     if args.model == "svm":
#         base = LinearSVC(C=args.svm_c, class_weight=class_wt, random_state=args.random_state)
#         return CalibratedClassifierCV(base, cv=3) if args.calibrate_svm else base

#     if args.model == "xgb":
#         if not HAS_XGB:
#             raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
#         return XGBClassifier(
#             n_estimators=400, max_depth=8, learning_rate=0.15,
#             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
#             tree_method="hist", eval_metric="mlogloss",
#             objective="multi:softprob", n_jobs=1, random_state=args.random_state
#         )

#     if args.model == "vote":
#         if not HAS_XGB:
#             raise RuntimeError("voting needs xgboost. Run: pip install xgboost")
#         lr = LogisticRegression(
#             max_iter=4000, solver="saga", multi_class="multinomial",
#             C=args.lr_c, class_weight=class_wt, random_state=args.random_state
#         )
#         svm = LinearSVC(C=args.svm_c, class_weight=class_wt, random_state=args.random_state)
#         svm = CalibratedClassifierCV(svm, cv=3)  # soft voting
#         xgb = XGBClassifier(
#             n_estimators=400, max_depth=8, learning_rate=0.15,
#             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
#             tree_method="hist", eval_metric="mlogloss",
#             objective="multi:softprob", n_jobs=1, random_state=args.random_state
#         )
#         return VotingClassifier(
#             estimators=[("lr", lr), ("svm", svm), ("xgb", xgb)],
#             voting="soft", n_jobs=None, flatten_transform=True
#         )
#     raise ValueError("unknown model")


# def param_distributions(model_name):
#     if model_name == "logreg":
#         return {"clf__C": np.logspace(-1, 1.2, 20)}
#     if model_name == "svm":
#         return {"clf__C": np.logspace(-1, 1.2, 20)}
#     if model_name == "xgb":
#         return {
#             "clf__n_estimators": [300, 400, 600],
#             "clf__max_depth": [6, 8, 10],
#             "clf__learning_rate": [0.07, 0.1, 0.15, 0.2],
#             "clf__subsample": [0.8, 0.9, 1.0],
#             "clf__colsample_bytree": [0.8, 0.9, 1.0],
#             "clf__reg_lambda": [0.5, 1.0, 2.0],
#         }
#     if model_name == "vote":
#         return {
#             "clf__lr__C": np.logspace(-1, 1.2, 10),
#             "clf__svm__base_estimator__C": np.logspace(-1, 1.2, 10),
#             "clf__xgb__n_estimators": [300, 400, 600],
#             "clf__xgb__learning_rate": [0.07, 0.1, 0.15],
#             "clf__xgb__max_depth": [6, 8, 10],
#         }
#     return {}


# def build_pipeline(args):
#     feats = make_features()
#     clf = make_clf(args)
#     return Pipeline(steps=[("tfidf", feats), ("clf", clf)])


# def maybe_tune(pipe, X, y, args):
#     if args.tune == "none":
#         return pipe
#     n_iter = args.tune_iter if args.tune_iter else (20 if args.tune == "light" else 60)
#     dist = param_distributions(args.model)
#     if not dist:
#         return pipe
#     print(f"Tuning ({args.tune}) with n_iter={n_iter} ...")
#     rs = RandomizedSearchCV(
#         pipe, dist, n_iter=n_iter, cv=3, n_jobs=1, verbose=1,
#         scoring="accuracy", random_state=args.random_state
#     )
#     rs.fit(X, y)
#     print("Best params:", rs.best_params_)
#     print("Best CV acc:", rs.best_score_)
#     return rs.best_estimator_


# def main():
#     args = parse_args()
#     data_path = Path(args.data)

#     print(f"Loading data from : {data_path}")
#     df = load_data(data_path, args.min_words, args.min_class_count)
#     total = len(df)
#     print(f"Total rows after cleaning: {total}")

#     SAMPLE = int(args.sample)
#     if SAMPLE <= 0 or SAMPLE >= total:
#         print("Using FULL dataset (no sampling).")
#     else:
#         frac = SAMPLE / total
#         print(f"Sampling approx: {SAMPLE} rows (~{frac:.2%}) ...")
#         df = (df.groupby("genre", group_keys=False)
#               .apply(lambda g: g.sample(max(1, int(len(g)*frac)), random_state=args.random_state))
#               .reset_index(drop=True))
#     print(f"Using {len(df)} rows after stratified sample.")

#     if args.top_k and args.top_k > 0:
#         topk = df["genre"].value_counts().nlargest(args.top_k).index
#         df = df[df["genre"].isin(topk)].reset_index(drop=True)
#         print(f"Kept top-{args.top_k} genres. Rows now: {len(df)}")
#     if args.balance:
#         counts = df["genre"].value_counts()
#         target = counts.min() if args.per_class_cap == 0 else min(args.per_class_cap, counts.min())
#         df = (df.groupby("genre", group_keys=False)
#                 .apply(lambda g: g.sample(n=min(len(g), target), random_state=args.random_state))
#                 .reset_index(drop=True))
#         print(f"Balanced sampling done. ~{target} per class. Rows now: {len(df)}")

#     classes = sorted(df["genre"].unique())
#     print(f"Genres: {classes}")

#     le = LabelEncoder()
#     y = le.fit_transform(df["genre"])
#     X = df["plot"]

#     base_pipe = build_pipeline(args)
#     pipe = maybe_tune(base_pipe, X, y, args)

#     print("\n=== 5-fold CV (memory safe mode) ===")
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
#     acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
#     f1_macro = cross_val_score(pipe, X, y, cv=skf,
#                                scoring=make_scorer(f1_score, average="macro"), n_jobs=1)
#     f1_weighted = cross_val_score(pipe, X, y, cv=skf,
#                                   scoring=make_scorer(f1_score, average="weighted"), n_jobs=1)

#     print("\n=== 5-fold CV ===")
#     print(f"Accuracy:     mean={acc_scores.mean():.3f}  (± {acc_scores.std():.3f})")
#     print(f"Macro F1:     mean={f1_macro.mean():.3f}    (± {f1_macro.std():.3f})")
#     print(f"Weighted F1:  mean={f1_weighted.mean():.3f} (± {f1_weighted.std():.3f})")

#     pipe.fit(X, y)
#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump({"pipeline": pipe, "labeler": le, "classes": classes}, MODEL_PATH)
#     print(f"\nSaved model to: {MODEL_PATH}")


# if __name__ == "__main__":
#     main()
