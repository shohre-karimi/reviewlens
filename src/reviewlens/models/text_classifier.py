"""A trained model bundled with everything needed to classify raw text.

Experiments work on feature matrices, but the app and the agent receive a raw review.
``TextClassifier`` runs the same steps as training (normalize -> TF-IDF -> [LSA] -> model),
so a prediction at inference time sees exactly the same kind of input the model was trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from reviewlens.models.base import Classifier, FloatArray
from reviewlens.preprocessing import normalize_texts


@dataclass
class TextClassifier:
    method: str
    labels: list[str]
    vectorizer: TfidfVectorizer
    model: Classifier
    lsa: Pipeline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. label fraction, seed, metrics

    def _features(self, texts: Sequence[str]) -> Any:
        matrix = self.vectorizer.transform(normalize_texts(texts))
        return self.lsa.transform(matrix) if self.lsa is not None else matrix

    def predict_proba(self, texts: Sequence[str]) -> FloatArray:
        return self.model.predict_proba(self._features(texts))

    def predict_labels(self, texts: Sequence[str]) -> list[str]:
        ids = np.argmax(self.predict_proba(texts), axis=1)
        return [self.labels[i] for i in ids]


def save_text_classifier(classifier: TextClassifier, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)
    return path


def load_text_classifier(path: Path) -> TextClassifier:
    """Load a saved classifier. Only load files this project created: joblib uses pickle."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}. Train one with `reviewlens train`.")
    loaded = joblib.load(path)
    if not isinstance(loaded, TextClassifier):
        raise TypeError(f"{path} does not contain a TextClassifier")
    return loaded
