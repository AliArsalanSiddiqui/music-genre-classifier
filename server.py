import os
import math
import json
import shutil
import tempfile
from collections import Counter

import numpy as np
import librosa
import tensorflow.keras as keras
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "weights/cnn_weights.keras"
MAPPING     = ["blues","classical","country","disco","hiphop",
               "jazz","metal","pop","reggae","rock"]

GENRE_INFO = {
    "blues":     {"sub": "Delta / Chicago Blues", "mood": "Soulful", "energy": "Medium", "dance": "Low", "acoustic": "High",
                  "artists": ["B.B. King","Muddy Waters","Robert Johnson","Howlin Wolf"],
                  "prod": "Raw, guitar-driven sound with expressive vocals and simple rhythm sections.",
                  "insights": ["Strong use of blue notes and pentatonic scales","Call-and-response vocal patterns","12-bar chord progressions dominate"]},
    "classical": {"sub": "Orchestral / Instrumental", "mood": "Sophisticated", "energy": "Variable", "dance": "Low", "acoustic": "Very High",
                  "artists": ["Mozart","Beethoven","Bach","Chopin"],
                  "prod": "Rich orchestration, dynamic contrast, and complex harmonic structures.",
                  "insights": ["Wide dynamic range from pianissimo to fortissimo","Complex polyphonic textures","Formal structures like sonata or symphony form"]},
    "country":   {"sub": "Contemporary Country", "mood": "Heartfelt", "energy": "Medium", "dance": "Medium", "acoustic": "High",
                  "artists": ["Johnny Cash","Dolly Parton","Hank Williams","Garth Brooks"],
                  "prod": "Twangy guitars, fiddles, and storytelling lyrics with a warm mix.",
                  "insights": ["Narrative storytelling is central","Steel guitar and fiddle are signature instruments","Strong emphasis on lyrical content"]},
    "disco":     {"sub": "Classic Disco / Dance", "mood": "Euphoric", "energy": "High", "dance": "Very High", "acoustic": "Low",
                  "artists": ["Donna Summer","Bee Gees","Gloria Gaynor","Chic"],
                  "prod": "Four-on-the-floor beats, lush strings, and funky bass lines built for the dance floor.",
                  "insights": ["Four-on-the-floor kick drum pattern","Syncopated bass lines drive the groove","Lush orchestral arrangements over electronic rhythm"]},
    "hiphop":    {"sub": "Hip-Hop / Rap", "mood": "Confident", "energy": "High", "dance": "High", "acoustic": "Low",
                  "artists": ["Kendrick Lamar","Jay-Z","Nas","Tupac"],
                  "prod": "Sample-based or synthesized beats with rhythmic vocal delivery and heavy bass.",
                  "insights": ["Rhythmic spoken word over sampled beats","Heavy low-frequency bass presence","Syncopation and off-beat accents are common"]},
    "jazz":      {"sub": "Modern Jazz / Swing", "mood": "Smooth", "energy": "Medium", "dance": "Medium", "acoustic": "High",
                  "artists": ["Miles Davis","John Coltrane","Duke Ellington","Louis Armstrong"],
                  "prod": "Complex chords, improvisation, and interplay between instruments.",
                  "insights": ["Extended chord voicings (7ths, 9ths, 13ths)","Improvisation is a core element","Swing rhythm feel with syncopated accents"]},
    "metal":     {"sub": "Heavy / Thrash Metal", "mood": "Intense", "energy": "Very High", "dance": "Low", "acoustic": "Low",
                  "artists": ["Metallica","Black Sabbath","Iron Maiden","Slayer"],
                  "prod": "Heavily distorted guitars, fast tempos, and powerful drumming with aggressive vocals.",
                  "insights": ["High-gain distorted guitar tones","Fast double-kick drum patterns","Down-tuned guitars for heavier sound"]},
    "pop":       {"sub": "Contemporary Pop", "mood": "Upbeat", "energy": "High", "dance": "High", "acoustic": "Medium",
                  "artists": ["Michael Jackson","Madonna","Taylor Swift","Ariana Grande"],
                  "prod": "Polished production with catchy hooks, verse-chorus structure, and wide stereo mix.",
                  "insights": ["Verse-chorus-bridge song structure","Heavy use of compression for loudness","Catchy melodic hooks are central"]},
    "reggae":    {"sub": "Roots Reggae", "mood": "Laid-back", "energy": "Medium", "dance": "Medium", "acoustic": "Medium",
                  "artists": ["Bob Marley","Peter Tosh","Burning Spear","Toots Hibbert"],
                  "prod": "Off-beat guitar skank, heavy bass, and relaxed tempo with conscious lyrics.",
                  "insights": ["Emphasis on off-beat (skank) guitar rhythm","Bass guitar carries the melodic weight","Conscious and spiritual lyrical themes"]},
    "rock":      {"sub": "Classic / Alternative Rock", "mood": "Energetic", "energy": "High", "dance": "Medium", "acoustic": "Medium",
                  "artists": ["Led Zeppelin","The Beatles","Nirvana","AC/DC"],
                  "prod": "Guitar-driven with powerful drums, bass, and dynamic song structures.",
                  "insights": ["Electric guitar riffs are the primary hook","Loud-quiet-loud dynamic contrast","Standard verse-chorus with guitar solos"]},
}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading CNN model…")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded ✓")

app = FastAPI()

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_mfccs(file_path, segment_duration=3, n_mfcc=13, n_fft=2048, hop_length=512, sample_rate=22050):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    samples_per_segment   = sample_rate * segment_duration
    expected_vector_length = math.ceil(samples_per_segment / hop_length)
    mfccs = []
    num_segments = int(len(signal) / samples_per_segment)
    for s in range(num_segments):
        start  = samples_per_segment * s
        finish = start + samples_per_segment
        if finish > len(signal):
            break
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr,
                                     n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == expected_vector_length:
            mfccs.append(mfcc.tolist())
    return np.array(mfccs), signal, sr

def extract_audio_features(signal, sr):
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
    sc       = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
    chroma   = librosa.feature.chroma_cqt(y=signal, sr=sr)
    key_idx  = int(np.argmax(np.mean(chroma, axis=1)))
    keys     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    harm, perc = librosa.effects.hpss(signal)
    h_ratio  = float(np.mean(np.abs(harm))  / (np.mean(np.abs(signal)) + 1e-8))
    p_ratio  = float(np.mean(np.abs(perc))  / (np.mean(np.abs(signal)) + 1e-8))
    rms      = librosa.feature.rms(y=signal)[0]
    dyn      = float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-8) + 1e-8))
    return {
        "tempo_bpm":         round(float(np.atleast_1d(tempo)[0]), 1),
        "estimated_key":     keys[key_idx],
        "duration_seconds":  round(len(signal) / sr, 1),
        "spectral_centroid": round(sc, 1),
        "harmonic_ratio":    round(h_ratio, 3),
        "percussive_ratio":  round(p_ratio, 3),
        "dynamic_range_db":  round(dyn, 1),
    }

# ── Classify endpoint ─────────────────────────────────────────────────────────
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        mfccs, signal, sr = extract_mfccs(tmp_path)
        if len(mfccs) == 0:
            raise HTTPException(status_code=400, detail="Audio too short — need at least 3 seconds.")

        # CNN expects (batch, time, mfcc, 1)
        X = mfccs[..., np.newaxis]
        preds  = model.predict(X, verbose=0)
        probs  = tf.nn.softmax(preds, axis=-1).numpy()

        # Per-segment winner, then majority vote
        seg_classes = np.argmax(probs, axis=1)
        counts      = Counter(seg_classes)
        top_idx     = counts.most_common(1)[0][0]
        genre       = MAPPING[top_idx]

        # Average probability across segments for each class
        mean_probs = np.mean(probs, axis=0)
        confidence = round(float(mean_probs[top_idx]) * 100, 1)

        genre_probabilities = {
            MAPPING[i]: round(float(mean_probs[i]) * 100, 1)
            for i in np.argsort(mean_probs)[::-1]
        }

        audio_features = extract_audio_features(signal, sr)
        info = GENRE_INFO.get(genre, {})

        return JSONResponse({
            "method":              "ast-model",
            "primary_genre":       genre.capitalize(),
            "sub_genre":           info.get("sub", ""),
            "confidence":          confidence,
            "mood":                info.get("mood", ""),
            "energy_level":        info.get("energy", ""),
            "danceability":        info.get("dance", ""),
            "acousticness":        info.get("acoustic", ""),
            "genre_probabilities": genre_probabilities,
            "audio_features":      audio_features,
            "key_insights":        info.get("insights", []),
            "similar_artists":     info.get("artists", []),
            "production_style":    info.get("prod", ""),
            "brief_summary":       f"This track shows strong characteristics of {genre} music with {info.get('mood','').lower()} qualities and {info.get('energy','').lower()} energy levels.",
        })

    finally:
        os.unlink(tmp_path)
