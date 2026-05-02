## 🎵 AI Music Genre Classifier
   
An end-to-end AI-powered music genre classification web app built with a CNN trained on the GTZAN dataset, served via FastAPI, with a modern drag-and-drop frontend.
    
link to deployed webservice: https://music-genre-classifier-j13y.onrender.com

---

## 🧠 How It Works

1. User uploads an audio file (MP3, WAV, FLAC, OGG, M4A)
2. Backend extracts MFCC features using `librosa`
3. CNN model predicts genre probabilities across segments
4. Majority voting determines the final genre
5. Results + audio features + metadata returned to frontend

---

## 📁 Project Structure

```
music-genre-classifier/
├── main.ipynb   # Training notebook
├── server.py                # FastAPI backend
├── index.html               # Frontend UI
├── requirements.txt         # Python dependencies
├── processed.json           # Generated after training
├── weights/
│   └── cnn_weights.keras    # Generated after training
└── genres_original/         # GTZAN dataset
    ├── blues/
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
```

---

## ⚙️ Local Setup

### 1. Install Python 3.11
Download from [python.org](https://python.org) — check **Add to PATH** during install.

### 2. Install dependencies
Open Command Prompt and run:
```bash
py -3.11 -m pip install matplotlib librosa numpy scikit-learn tensorflow fastapi uvicorn python-multipart ipykernel
```

### 3. Register Jupyter kernel
```bash
py -3.11 -m ipykernel install --user --name py311 --display-name "Python 3.11"
```

### 4. Train the model
- Open `main.ipynb` in Jupyter
- Switch kernel to **Python 3.11**
- Update paths in the notebook:
  - `save_mfcc(...)` → use `"genres_original"` and `"processed.json"`
  - `load_data(...)` → use `"processed.json"`
  - model save → `"weights/cnn_weights.keras"`
- Run **Section 2** (MFCC extraction) — generates `processed.json`
- Run **Section 4** (CNN) — trains model, saves `weights/cnn_weights.keras`

### 5. Start the server
```bash
cd path\to\music-genre-classifier
py -3.11 -m uvicorn server:app --reload
```

### 6. Open the app
Go to **http://localhost:8000** in your browser.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `tensorflow not found` | You're on Python 3.14 — switch to Python 3.11 |
| `ModuleNotFoundError` | Run `pip install <module>` then restart kernel |
| `jazz.00054.wav error` | Known corrupted GTZAN file — safely ignored |
| `index.html not found` | Put `index.html` in same folder as `server.py` |
| `Audio too short` | File must be at least 3 seconds long |

---

## 🚀 Deployment

See the deployment section in the project report PDF for full cloud deployment instructions (Render recommended).

---

## 📊 Model Performance

| Model | Test Accuracy | Epochs |
|-------|--------------|--------|
| Dense NN | ~66% | 100 |
| **CNN (selected)** | **~78%** | **30** |
| LSTM | ~72% | 30 |

---

## 📚 Dataset

[GTZAN Music Genre Classification (dataset original + processed with wieghts)](https://www.kaggle.com/datasets/aliaralan67/gtzan-dataset-original-processed-with-wieghts) — 1000 audio clips, 10 genres, 30 seconds each.
     
