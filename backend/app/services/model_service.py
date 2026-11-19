"""
Model service: loads the pretrained DistilBERT question-answering model once.

Model: distilbert/distilbert-base-cased-distilled-squad
    A DistilBERT model (a smaller, faster, distilled version of BERT) that was
    fine-tuned by Hugging Face on the SQuAD v1.1 reading-comprehension dataset.
    It is an EXTRACTIVE model: given a question and a context paragraph it
    predicts the start and end position of the answer span inside the context.
    It cannot generate text that is not present in the context.

No training or fine-tuning happens in this project; the weights are used as-is.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from app.exceptions import ModelInferenceError, ModelNotReadyError

logger = logging.getLogger(__name__)


class ModelStatus:
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class ModelService:
    """Holds the tokenizer, model and Hugging Face QA pipeline."""

    def __init__(self, model_name: str, device_preference: str = "auto", max_seq_len: int = 512):
        self.model_name = model_name
        self.device_preference = device_preference
        self.max_seq_len = max_seq_len

        self.tokenizer: Any = None
        self.model: Any = None
        self.qa_pipeline: Any = None
        self.device: str = "unknown"
        self.device_name: Optional[str] = None
        self.status: str = ModelStatus.NOT_LOADED
        self.error: Optional[str] = None
        self.load_seconds: Optional[float] = None

        # The pipeline is not guaranteed to be thread-safe and FastAPI runs
        # synchronous endpoints in a thread pool, so inference is serialised.
        self._inference_lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self.status == ModelStatus.LOADED

    def _select_device(self) -> int:
        """Return the pipeline device index: 0 for the first GPU, -1 for CPU."""
        import torch

        cuda_available = torch.cuda.is_available()
        if self.device_preference == "cuda" and not cuda_available:
            logger.warning("DEVICE=cuda was requested but CUDA is not available; using CPU.")
        use_cuda = cuda_available and self.device_preference in ("auto", "cuda")

        if use_cuda:
            self.device = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            return 0
        self.device = "cpu"
        self.device_name = None
        return -1

    def load(self) -> None:
        """
        Download (first run only) and load the tokenizer, model and pipeline.

        Failures are recorded instead of crashing the server, so the health
        endpoint can report a helpful error message.
        """
        self.status = ModelStatus.LOADING
        self.error = None
        started = time.perf_counter()
        logger.info("Loading QA model '%s' ...", self.model_name)
        try:
            # Imported here so that importing this module (e.g. in tests) is cheap.
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

            device_index = self._select_device()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.eval()  # inference mode (disables dropout)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_index,
            )
        except Exception as exc:
            self.status = ModelStatus.FAILED
            # Public message (shown by /api/health). Full details, which may
            # contain local file paths, are written to the server log only.
            self.error = (
                f"Could not load the pretrained model '{self.model_name}' ({type(exc).__name__}). "
                "On the first run the model (~260 MB) is downloaded from the Hugging Face Hub, "
                "so check your internet connection, or make sure the model is already cached. "
                "See the server log for details."
            )
            logger.exception("Model loading failed")
            return

        self.load_seconds = round(time.perf_counter() - started, 2)
        self.status = ModelStatus.LOADED
        logger.info(
            "Model loaded on %s%s in %.2fs",
            self.device,
            f" ({self.device_name})" if self.device_name else "",
            self.load_seconds,
        )

    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        if not self.is_loaded:
            if self.status == ModelStatus.FAILED:
                raise ModelNotReadyError(
                    "The question-answering model failed to load, so questions cannot be "
                    "answered. Check the server logs and /api/health for details."
                )
            raise ModelNotReadyError()

    def count_tokens(self, text: str) -> int:
        """Number of DistilBERT tokens in `text` (without special tokens)."""
        self.ensure_ready()
        return len(self.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def predict(self, question: str, context: str, max_answer_len: int = 30) -> Dict[str, Any]:
        """
        Run extractive QA on one (question, context) pair.

        Returns the pipeline output:
            {"answer": str, "score": float, "start": int, "end": int}
        where start/end are character offsets of the answer inside `context`.
        """
        self.ensure_ready()
        try:
            with self._inference_lock:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=max_answer_len,
                    max_seq_len=self.max_seq_len,
                    handle_impossible_answer=False,
                    top_k=1,
                )
        except Exception as exc:
            logger.exception("QA inference failed")
            raise ModelInferenceError() from exc

        # Depending on the input the pipeline may return a dict or a list of one dict.
        if isinstance(result, list):
            result = result[0] if result else {"answer": "", "score": 0.0, "start": 0, "end": 0}
        return {
            "answer": str(result.get("answer", "")),
            "score": float(result.get("score", 0.0)),
            "start": int(result.get("start", 0)),
            "end": int(result.get("end", 0)),
        }

    def predict_candidates(
        self,
        question: str,
        context: str,
        max_answer_len: int = 50,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `top_k` candidate answer spans from `context`.

        When running with mocked tests (`qa_pipeline is None`), delegates to
        `self.predict(...)` so unit-test overrides continue to work seamlessly.
        """
        self.ensure_ready()
        if self.qa_pipeline is None:
            return [self.predict(question, context, max_answer_len=max_answer_len)]

        try:
            with self._inference_lock:
                results = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=max_answer_len,
                    max_seq_len=self.max_seq_len,
                    handle_impossible_answer=False,
                    top_k=top_k,
                )
        except Exception as exc:
            logger.exception("QA inference failed")
            raise ModelInferenceError() from exc

        if isinstance(results, dict):
            results = [results]
        return [
            {
                "answer": str(item.get("answer", "")),
                "score": float(item.get("score", 0.0)),
                "start": int(item.get("start", 0)),
                "end": int(item.get("end", 0)),
            }
            for item in (results or [])
        ]

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_status": self.status,
            "model_loaded": self.is_loaded,
            "device": self.device,
            "device_name": self.device_name,
            "model_error": self.error,
        }
