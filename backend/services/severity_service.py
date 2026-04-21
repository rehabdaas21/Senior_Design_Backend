from pathlib import Path
import numpy as np
import joblib
import librosa
from engine_inputs_outputs import SeverityLevel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "HAIRCSM_UPLOAD" / "models"

SEV_MAP = {0: SeverityLevel.MILD, 1: SeverityLevel.MODERATE, 2: SeverityLevel.SEVERE}

_clf = None
_tfidf = None
_models_attempted = False

def _load_models():
    global _clf, _tfidf, _models_attempted
    if _models_attempted:
        return _clf, _tfidf

    _models_attempted = True
    try:
        _clf = joblib.load(MODELS_DIR / "fused_severity_model.pkl")
        _tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    except Exception:
        _clf = None
        _tfidf = None
    return _clf, _tfidf

def _extract_audio_features(wav_path: str) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)  # (64, T)
    target_T = 3000
    T = mfcc.shape[1]
    if T < target_T:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_T - T)), mode="constant")
    else:
        mfcc = mfcc[:, :target_T]
    return mfcc.astype(np.float32).reshape(1, -1)

def predict_severity(audio_file: str = None, text: str = None) -> SeverityLevel | None:
    """
    Predicts aphasia severity using fused audio (MFCC) + text (TF-IDF) features.
    Falls back to MODERATE if neither audio nor text is provided.
    """
    if audio_file is None and not text:
        return None

    try:
        clf, tfidf = _load_models()
        if clf is None or tfidf is None:
            return None
    except Exception:
        return None

    # Audio features (192000)
    if audio_file:
        audio_feats = _extract_audio_features(audio_file)
    else:
        audio_feats = np.zeros((1, 192000), dtype=np.float32)

    # Text features (500)
    text_input = text if text else ""
    text_feats = tfidf.transform([text_input]).toarray().astype(np.float32)

    # Fuse and predict
    X = np.concatenate([audio_feats, text_feats], axis=1)
    pred = int(clf.predict(X)[0])
    return SEV_MAP.get(pred, SeverityLevel.MODERATE)
