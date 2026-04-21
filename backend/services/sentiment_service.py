import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    DistilBertForSequenceClassification, DistilBertTokenizerFast
)
from engine_inputs_outputs import EmotionState

BASE_DIR          = Path(__file__).resolve().parent.parent
FUSION_CHECKPOINT = BASE_DIR / "models" / "fusion_model_final.pt"
DISTILBERT_PATH   = str(BASE_DIR / "models" / "text_sentiment_model")
SAMPLE_RATE       = 16000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.compress = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256),
        )

    def forward(self, input_values, attention_mask):
        hidden = self.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).float()
        frame_lengths = (lengths / 320).ceil().long().clamp(max=hidden.size(1))
        mask = torch.zeros(hidden.size(0), hidden.size(1), device=hidden.device)
        for i, l in enumerate(frame_lengths):
            mask[i, :l] = 1.0
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.compress(pooled)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(258, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 32),  nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, audio_emb, text_probs):
        gate     = self.text_gate(text_probs)
        combined = torch.cat([audio_emb, text_probs * gate], dim=1)
        return self.classifier(combined), gate


_models = None
_models_attempted = False
_models_load_error: str | None = None

def _load_models():
    """
    Returns loaded models or `None` if models can't be loaded (e.g. no internet
    in restricted demo environment).
    """
    global _models, _models_attempted, _models_load_error
    if _models_attempted:
        return _models

    _models_attempted = True
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        distilbert = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_PATH
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        distilbert.eval().to(DEVICE)
        for param in distilbert.parameters():
            param.requires_grad = False

        ckpt = torch.load(str(FUSION_CHECKPOINT), map_location=DEVICE)
        audio_model = AudioModel().to(DEVICE)
        fusion_model = FusionModel().to(DEVICE)
        audio_model.load_state_dict(ckpt["audio_model"])
        fusion_model.load_state_dict(ckpt["fusion_model"])
        audio_model.eval()
        fusion_model.eval()

        _models = (processor, distilbert, tokenizer, audio_model, fusion_model)
        return _models
    except Exception as e:
        _models = None
        _models_load_error = str(e)
        return None


def predict_emotion(text: str, audio_file: str = None) -> EmotionState | None:
    """
    Predicts emotion from text (and optionally audio).
    If audio_file is provided, uses the full fusion model.
    If only text is provided, uses DistilBERT alone.
    """
    try:
        models = _load_models()
        if models is None:
            return None

        processor, distilbert, tokenizer, audio_model, fusion_model = models

        with torch.no_grad():
            inputs = tokenizer(
                text, max_length=128, truncation=True, padding=True, return_tensors="pt"
            ).to(DEVICE)
            text_probs = F.softmax(distilbert(**inputs).logits, dim=-1)

            if audio_file is not None:
                audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
                audio = audio[: SAMPLE_RATE * 600]
                audio_inputs = processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                audio_emb = audio_model(
                    audio_inputs["input_values"].to(DEVICE),
                    audio_inputs["attention_mask"].to(DEVICE),
                )
                logits, _ = fusion_model(audio_emb, text_probs)
                probs = F.softmax(logits, dim=-1)
                is_positive = probs[0, 1].item() > probs[0, 0].item()
            else:
                # text-only: use DistilBERT directly
                is_positive = text_probs[0, 1].item() > text_probs[0, 0].item()

        return EmotionState.POSITIVE if is_positive else EmotionState.FRUSTRATED
    except Exception:
        return None
