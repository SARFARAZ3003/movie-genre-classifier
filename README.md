# Movie Script Genre Classifier (Starter)

A starter repository for a movie-genre classification pipeline that predicts a movie's genre from its script/plot text using TF-IDF and classical ML models.

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
Project Structure
pgsql
Copy code
movie-genre-classifier/
├── data/
│   └── movies.csv                # add dataset CSV with columns: title, plot, genre
├── model/
│   ├── train.py                  # training script (TF-IDF + classifier)
│   └── model.joblib              # produced model artifact after training
├── app/
│   ├── predict.py                # prediction wrapper
│   └── app.py                    # Gradio/Flask app for inference
├── notebooks/                    # optional experiments and EDA
├── requirements.txt
└── README.md
Quick Usage
Install dependencies: pip install -r requirements.txt

Train (example): python model/train.py

Run the demo UI: python app/app.py

Notes & Next Steps
The training script should: load data, preprocess text (tokenize, lowercase, remove stopwords), vectorize with TF-IDF, train model(s) (LinearSVC / LogisticRegression), evaluate (accuracy, precision/recall/F1), and dump model.joblib.

The app folder wraps the model for quick inference (e.g., Gradio).

Future improvements: transformer embeddings (BERT), larger dataset, hyperparameter search, deployment.

Contact
For issues or questions, contact: sarfaraz.hussain.work@gmail.com