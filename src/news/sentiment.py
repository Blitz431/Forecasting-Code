"""Phase 5: FinBERT-based sentiment scoring for financial news.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial phrases.
Labels: positive, negative, neutral.

The scorer is a singleton-friendly class. Instantiate once and reuse
across multiple batches to avoid reloading the model weights repeatedly.
GPU (CUDA) is used automatically when available.
"""

from __future__ import annotations

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_DEFAULT_MODEL = "ProsusAI/finbert"

# Maps FinBERT output label -> numeric direction (-1 / 0 / +1)
_DIRECTION: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}


class FinBERTScorer:
    """Scores financial text sentiment using FinBERT.

    Args:
        model_name: HuggingFace model ID. Defaults to ``ProsusAI/finbert``.

    Example::

        scorer = FinBERTScorer()
        results = scorer.score(["AAPL beats earnings estimates", "SEC probes fraud"])
        # [{'label': 'positive', 'score': 0.92, 'confidence': 0.92},
        #  {'label': 'negative', 'score': -0.87, 'confidence': 0.87}]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        # Lazy imports so the module can be imported without torch installed
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT ({model_name}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # id2label: {0: 'positive', 1: 'negative', 2: 'neutral'} (FinBERT default)
        self._id2label: dict[int, str] = self.model.config.id2label

    def score(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Score a list of financial texts.

        Args:
            texts: Plaintext strings (headlines, summaries, etc.).
            batch_size: Number of texts to process per GPU/CPU pass.

        Returns:
            List of dicts (same length as ``texts``) with:
            - ``label``: "positive", "negative", or "neutral"
            - ``score``: float in [-1, 1] — confidence * direction
            - ``confidence``: raw softmax probability for the winning class
        """
        if not texts:
            return []

        torch = self._torch
        results: list[dict] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for prob_vec in probs:
                label_probs = {self._id2label[idx]: p for idx, p in enumerate(prob_vec)}
                best_label = max(label_probs, key=label_probs.__getitem__)
                confidence = label_probs[best_label]
                numeric = round(_DIRECTION.get(best_label, 0.0) * confidence, 4)
                results.append({
                    "label": best_label,
                    "score": numeric,
                    "confidence": round(confidence, 4),
                })

        return results

    def score_single(self, text: str) -> dict:
        """Convenience wrapper to score a single text string."""
        return self.score([text])[0]
