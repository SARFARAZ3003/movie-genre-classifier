import pandas as pd
from pathlib import Path
import argparse

# def load_guess(path :Path) -> pd.DataFrame:
#     tried=[]
#     candidates = [
#         dict(sep="\t", header=0),
#         dict(sep=",",header=0),
#         dict(sep="|",header=0),
#         dict(sep="\t",header=None),
#         dict(sep=",",header=None),
#         dict(sep="|",header=None)
#     ]

#     for opt in candidates:
#         try:
#             df =pd.read_csv(path, **opt, engine="python")
#             if df.shape[1] >= 2:
#                 return df
#             tried.append(opt)
#         except Exception:
#             tried.append(opt)
#             continue
#     raise ValueError(f"Could not parse file with common separators. Tried: {tried}")


# def normalize(df: pd.DataFrame) -> pd.DataFrame:

#     text_cols = [c for c in df.columns if str(c).lower() in ["plot","text","overview","synopsis","description","story"]]
#     label_cols = [c for c in df.columns if str(c).lower() in ["genre","label","labels","category","target","genres"]]

#     if not text_cols or not label_cols :
#         #Fall back to first two columns
#         df = df.rename(columns={df.columns[0]:"plot", df.columns[1]:"genre"})
#     else :
#         df = df[[text_cols[0], label_cols[0]]].rename(columns={text_cols[0]:"plot", label_cols[0]:"genre"})

#     df["plot"] = df["plot"].fillna("").astype(str).str.strip()
#     df["genre"] = df["genre"].fillna("").astype(str).str.strip()

#     df["genre"] = df["genre"].str.split(r"[|,/]").str[0].str.strip()

#     df = df[(df["plot"]!="") & (df["genre"]!="")]
#     df = df.drop_duplicates(subset=["plot"]).reset_index(drop=True)
#     return df[["plot","genre"]]


def parse_file(path: Path) -> pd.DataFrame:
    rows =[]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split(":::")
            if len(parts) < 4:
                continue
            
            parts = [p.strip() for p in parts]
            #Format: id ::: title(year) ::: genre ::: plot
            _id = parts[0]
            title = parts[1]
            genre = parts[2]
            plot = ":::".join(parts[3:]).strip()
            rows.append((plot,genre))

        df = pd.DataFrame(rows, columns=["plot","genre"])
        #basic cleaning
        df = df[df["plot"] != ""]
        df = df[df["genre"] != ""]

        df["genre"] = df["genre"].str.split(r"[|,/]").str[0].str.strip()

        df = df.drop_duplicates(subset=["plot"]).reset_index(drop=True)
        return df



    


def main():
    ap = argparse.ArgumentParser(description="Convert Kaggle TXT to movies.csv (plot,genre)")
    ap.add_argument("--in", dest="inp", default=r"data\train_data.txt" , help="Path to Kaggle TXT")
    ap.add_argument("--out", dest="out", default=r"data\movies.csv", help="Output CSV Path")
    args= ap.parse_args()

    in_path =Path(args.inp)
    out_path =Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_file(in_path)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(df)}  cols={list(df.columns)}")   


if __name__=="__main__":
    main() 