from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, List

import re


_NEGATION_HINTS = {
    "not",
    "never",
    "no",
    "none",
    "deny",
    "denies",
    "denied",
    "false",
    "incorrect",
    "invalid",
    "contradict",
    "contradiction",
    "without",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    z = sum(exps)
    if z <= 0:
        return [0.0 for _ in exps]
    return [x / z for x in exps]


@dataclass
class StanceResult:
    label: str
    confidence: float
    support_prob: float
    refute_prob: float
    neutral_prob: float
    source: str


class NLIStanceClassifier:
    """
    Natural language inference stance classifier with fallback.

    Preferred model: cross-encoder/nli-deberta-v3-base
    Labels: support / refute / neutral
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        max_length: int = 512,
        neutral_threshold: float = 0.45,
    ):
        self.model_name = model_name
        self.max_length = max(64, int(max_length))
        self.neutral_threshold = max(0.0, min(1.0, float(neutral_threshold)))

        self._available: bool | None = None
        self._load_error: str = ""

        self._tokenizer = None
        self._model = None
        self._torch = None

    def _lazy_init(self) -> None:
        if self._available is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._available = True
            self._load_error = ""
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._available = False
            self._load_error = str(exc)
            self._tokenizer = None
            self._model = None
            self._torch = None

    @property
    def available(self) -> bool:
        self._lazy_init()
        return bool(self._available)

    @property
    def load_error(self) -> str:
        self._lazy_init()
        return self._load_error

    def _decode_probs(self, logits: List[float]) -> Dict[str, float]:
        probs = _softmax(logits)

        id2label = {}
        if self._model is not None and getattr(self._model, "config", None) is not None:
            id2label = getattr(self._model.config, "id2label", {}) or {}

        raw: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            label = str(id2label.get(idx, str(idx))).lower()
            raw[label] = float(p)

        entailment = 0.0
        contradiction = 0.0
        neutral = 0.0
        for label, p in raw.items():
            if "entail" in label:
                entailment += p
            elif "contradict" in label:
                contradiction += p
            elif "neutral" in label:
                neutral += p

        # fallback for unknown label names
        if entailment + contradiction + neutral <= 0.0 and len(probs) >= 3:
            # common order for MNLI-like heads: contradiction, neutral, entailment
            contradiction, neutral, entailment = probs[0], probs[1], probs[2]

        z = max(1e-9, entailment + contradiction + neutral)
        return {
            "entailment": entailment / z,
            "contradiction": contradiction / z,
            "neutral": neutral / z,
        }

    def _to_stance(self, probs: Dict[str, float], source: str) -> StanceResult:
        support = max(0.0, min(1.0, float(probs.get("entailment", 0.0))))
        refute = max(0.0, min(1.0, float(probs.get("contradiction", 0.0))))
        neutral = max(0.0, min(1.0, float(probs.get("neutral", 0.0))))

        label = "neutral"
        confidence = neutral
        if support >= refute and support >= neutral:
            label = "support"
            confidence = support
        elif refute >= support and refute >= neutral:
            label = "refute"
            confidence = refute

        if label != "neutral" and confidence < self.neutral_threshold:
            label = "neutral"
            confidence = max(confidence, neutral)

        return StanceResult(
            label=label,
            confidence=float(confidence),
            support_prob=support,
            refute_prob=refute,
            neutral_prob=neutral,
            source=source,
        )

    def _predict_with_model(self, claim_text: str, evidence_text: str) -> StanceResult:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None

        inputs = self._tokenizer(
            claim_text,
            evidence_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits_tensor = self._model(**inputs).logits[0]
        logits = [float(x) for x in logits_tensor.tolist()]
        probs = self._decode_probs(logits)
        return self._to_stance(probs, source="nli_model")

    def _predict_heuristic(self, claim_text: str, evidence_text: str) -> StanceResult:
        c_tokens = set(_tokenize(claim_text))
        e_tokens = set(_tokenize(evidence_text))
        overlap = 0.0
        if c_tokens:
            overlap = len(c_tokens.intersection(e_tokens)) / max(1, len(c_tokens))

        neg_hits = sum(1 for tok in e_tokens if tok in _NEGATION_HINTS)

        if overlap >= 0.45 and neg_hits > 0:
            label = "refute"
            refute = min(0.95, 0.55 + 0.4 * overlap)
            support = max(0.01, 0.2 * overlap)
            neutral = max(0.01, 1.0 - refute - support)
        elif overlap >= 0.30:
            label = "support"
            support = min(0.95, 0.5 + 0.45 * overlap)
            refute = 0.05 if neg_hits == 0 else 0.18
            neutral = max(0.01, 1.0 - support - refute)
        elif overlap >= 0.18 and neg_hits > 0:
            label = "refute"
            refute = min(0.85, 0.45 + 0.35 * overlap)
            support = 0.08
            neutral = max(0.01, 1.0 - refute - support)
        else:
            label = "neutral"
            neutral = min(0.95, 0.65 + (1.0 - overlap) * 0.3)
            support = max(0.02, overlap * 0.25)
            refute = max(0.02, overlap * 0.2)

        confidence = {"support": support, "refute": refute, "neutral": neutral}[label]
        return StanceResult(
            label=label,
            confidence=float(confidence),
            support_prob=float(support),
            refute_prob=float(refute),
            neutral_prob=float(neutral),
            source="heuristic",
        )

    def predict(self, claim_text: str, evidence_text: str) -> StanceResult:
        claim = " ".join((claim_text or "").strip().split())
        evidence = " ".join((evidence_text or "").strip().split())

        if not claim or not evidence:
            return StanceResult(
                label="neutral",
                confidence=1.0,
                support_prob=0.0,
                refute_prob=0.0,
                neutral_prob=1.0,
                source="empty",
            )

        if self.available:
            try:
                return self._predict_with_model(claim, evidence)
            except Exception:  # pragma: no cover - runtime dependency
                pass

        return self._predict_heuristic(claim, evidence)
