import os
import re
import sys
import pandas as pd

# ✅ Fix your project path automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_TXT  = os.path.join(DATA_DIR, "train_data.txt")     # Kaggle raw TXT
OUT_FULL   = os.path.join(DATA_DIR, "movies_full.csv")    # debug file
OUT_TRAIN  = os.path.join(DATA_DIR, "movies.csv")         # final file

MIN_WORDS  = 25  # drop small plots


# ---------------- CLEAN HELPERS ---------------- #

def normalize_spaces(s):
    s = str(s).strip()
    return re.sub(r"\s+", " ", s)


def extract_year_and_clean_title(title):
    """
    Extract (year) e.g. 'Avatar (2009)' → ('avatar', 2009)
    """
    t = str(title)
    m = re.search(r"\((\d{4})\)\s*$", t)
    year = int(m.group(1)) if m else None

    # remove the "(year)"
    t_clean = re.sub(r"\(\d{4}\)\s*$", "", t).strip()

    # normalized title
    t_norm = re.sub(r"\s+", " ", t_clean).lower().strip()

    return t_norm, year


def normalize_genre(g):
    g = str(g).strip().lower()

    # common cleanup
    g = g.replace(" ", "-")

    fix = {
        "sci fi": "sci-fi",
        "sci-fi": "sci-fi",
        "film noir": "film-noir",
        "game show": "game-show",
        "talk show": "talk-show",
        "reality tv": "reality-tv",
        "short film": "short",
    }

    return fix.get(g, g)


# ---------------- MAIN PROCESS ---------------- #

def main():
    print("📥 Reading:", INPUT_TXT)

    if not os.path.exists(INPUT_TXT):
        print("❌ train_data.txt NOT FOUND at:", INPUT_TXT)
        sys.exit(1)

    # read Kaggle train_data.txt
    df = pd.read_csv(
        INPUT_TXT,
        sep=":::",
        engine="python",
        header=None,
        names=["id", "title", "genre", "plot"],
        on_bad_lines="skip"
    )

    # basic clean
    for col in ["title", "genre", "plot"]:
        df[col] = df[col].astype(str).map(normalize_spaces)

    # extract year, normalized title
    tmp = df["title"].apply(extract_year_and_clean_title)
    df["title_norm"] = tmp.apply(lambda x: x[0])
    df["year"] = tmp.apply(lambda x: x[1])

    # normalize genre text
    df["genre"] = df["genre"].map(normalize_genre)

    # remove empty/short plots
    df["word_count"] = df["plot"].str.split().apply(len)
    before = len(df)
    df = df[df["word_count"] >= MIN_WORDS]
    print(f"🧹 Removed short plots (<{MIN_WORDS} words): {before - len(df)}")

    # drop exact plot duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["plot"])
    print(f"🧽 Removed exact duplicate plots: {before - len(df)}")

    # drop duplicates by (title_norm, year)
    before = len(df)
    df = (
        df.sort_values("word_count", ascending=False)
          .drop_duplicates(subset=["title_norm", "year"], keep="first")
    )
    print(f"🧽 Removed duplicate movies (title+year): {before - len(df)}")

    # Save two files
    df_full = df[["id", "title", "year", "genre", "plot"]]
    df_train = df[["plot", "genre"]]

    df_full.to_csv(OUT_FULL, index=False)
    df_train.to_csv(OUT_TRAIN, index=False)

    print("\n✅ CLEANING DONE!")
    print(f"   Final rows (unique movies): {len(df_train):,}")
    print(f"   Final genres: {sorted(df_train['genre'].unique())}")
    print(f"   Saved full file:   {OUT_FULL}")
    print(f"   Saved training file: {OUT_TRAIN}")
    print("🎯 You can now train your model on data/movies.csv")


if __name__ == "__main__":
    main()
