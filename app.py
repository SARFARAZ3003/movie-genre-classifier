# Step 5: Gradio UI placeholder
# TODO: Load model.joblib and wire predict + recommend into a Blocks interface.

import os, gradio as gr, pandas as pd
from model.predict import MovieGenrePredictor

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "model.joblib"))
_predictor = MovieGenrePredictor(model_path=MODEL_PATH)

def infer(text:str, k: int=3):
    t = (text or "").strip()
    if not t:
        return "", pd.DataFrame(columns=["rank","genre","probability"])
    res = _predictor.predict_proba_topk([t], k=k)[0]
    best = res[0]["label"] if res else ""
    df = pd.DataFrame([
        {"rank": i+1, "genre": r["label"], "probability": round(r["proba"], 4)}
        for i, r in enumerate(res)
    ])

    return best, df

with gr.Blocks(title="🎬 Movie Genre Classifier") as demo:
    gr.Markdown("# 🎬 Movie Genre Classifier\nTF-IDF + Logistic Regression")
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="Plot / Overview", lines=8, placeholder="Paste a movie plot...")
            k = gr.Slider(1,10, value=5, step=1, label="Top-K")
            btn = gr.Button("Predict")

        with gr.Column(scale=2):
            best = gr.Textbox(label="Predicted Genre", interactive=False)
            table = gr.DataFrame(
                headers =["rank","genre","probability"],
                datatype= ["number","str","number"],
                label="Top-K", interactive=False
            )
    
    btn.click(infer, [txt,k], [best, table])
    txt.submit(infer, [txt,k], [best, table])



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")