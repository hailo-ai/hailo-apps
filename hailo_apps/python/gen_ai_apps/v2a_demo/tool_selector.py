"""
Tool Selector Engine (Stage 2)

Interface:
    - __init__(vdevice): Initialize with VDevice for HEF inference
    - run(text) -> str: Classify user text, return tool name
    - close(): Clean up resources

Uses all-MiniLM-L6-v2 sentence-transformer compiled to HEF.
Classifies user input by computing embedding similarity against
pre-computed tool description embeddings.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from hailo_platform import VDevice, FormatType
from tools import TOOL_DESCRIPTIONS

try:
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import HAILO10H_ARCH, V2A_DEMO_APP
except ImportError:
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import HAILO10H_ARCH, V2A_DEMO_APP

logger = logging.getLogger("v2a_demo")

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
WORD_EMBEDDINGS_PATH = RESOURCES_DIR / "word_embeddings_weight.npy"
TOOL_EMBEDDINGS_CACHE_PATH = RESOURCES_DIR / "tool_embeddings_cache.npz"

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 128
TIMEOUT_MS = 10000
DEFAULT_TOOL = "none"

MODEL_NAME = "all_minilm_l6_v2"
EMBEDDING_INPUT_NAME = f"{MODEL_NAME}/input_layer1"
MASK_INPUT_NAME = f"{MODEL_NAME}/input_layer2"
OUTPUT_NAME = f"{MODEL_NAME}/normalization13"

# Per-tool minimum cosine similarity thresholds
TOOL_THRESHOLDS = {
    "get_weather": 0.35,
    "get_travel_time": 0.45,
    "control_led": 0.55,
    "system_check": 0.35,
    "explain_tools": 0.35,
    "data_storage": 0.40,
    "none": 0.0,
}


def _descriptions_hash() -> str:
    """Compute a hash of TOOL_DESCRIPTIONS for cache invalidation."""
    serialized = json.dumps(TOOL_DESCRIPTIONS, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def _hef_attention_mask(attention_mask: np.ndarray) -> np.ndarray:
    """Build the HEF mask input (batch, 1, seq, seq) from a (batch, seq) token mask.

    The HEF is compiled with the mask applied inside softmax, so it expects a
    multiplicative 0/1 mask
    """
    return np.repeat(attention_mask[:, None, None, :], MAX_SEQ_LEN, axis=2).astype(np.float32)


class ToolSelector:
    """Selects the appropriate tool using sentence-transformer embeddings on Hailo HEF."""

    def __init__(self, vdevice: VDevice):
        self._vdevice = vdevice
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self._text_embeddings = np.load(WORD_EMBEDDINGS_PATH)

        model_path = resolve_hef_path(
            hef_path="all_minilm_l6_v2",
            app_name=V2A_DEMO_APP,
            arch=HAILO10H_ARCH,
        )
        if model_path is None:
            raise RuntimeError("Failed to resolve HEF path for tool selector model 'all_minilm_l6_v2'")

        self._infer_model = vdevice.create_infer_model(str(model_path))
        for inp in self._infer_model.inputs:
            inp.set_format_type(FormatType.FLOAT32)
        for out in self._infer_model.outputs:
            out.set_format_type(FormatType.FLOAT32)

        self._configured_model = None
        self._bindings = None
        self._output_buffer = None
        self._tool_embeddings = None  # np.ndarray (num_descriptions, hidden_dim)
        self._tool_names = None       # list of tool names, one per description

    def _embed_text(self, text: str) -> np.ndarray:
        """Compute normalized embedding for a single text. Returns shape (1, hidden_dim)."""
        tokenized = self._tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="np",
        )
        input_ids = tokenized["input_ids"].astype(np.int64)
        attention_mask = tokenized["attention_mask"].astype(np.int64)

        # Word embedding lookup
        input_embeddings = self._text_embeddings[input_ids].astype(np.float32)

        attn_mask = _hef_attention_mask(attention_mask)

        # HEF inference
        self._bindings.input(EMBEDDING_INPUT_NAME).set_buffer(input_embeddings)
        self._bindings.input(MASK_INPUT_NAME).set_buffer(attn_mask)
        self._configured_model.run_async([self._bindings], lambda *_, **__: None).wait(TIMEOUT_MS)

        last_hidden = self._output_buffer  # (1, seq_len, hidden_dim)

        # Mean pooling (masked)
        mask = attention_mask[:, :, None].astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norm, a_min=1e-9, a_max=None)

    def _load_embeddings_from_cache(self) -> bool:
        """Try to load tool embeddings from disk cache. Returns True if successful."""
        current_hash = _descriptions_hash()
        TOOL_EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache = np.load(TOOL_EMBEDDINGS_CACHE_PATH, allow_pickle=True)
            if str(cache["hash"]) == current_hash:
                self._tool_embeddings = cache["embeddings"]
                self._tool_names = list(cache["names"])
                logger.info(f"Loaded tool embeddings from cache ({len(self._tool_names)} descriptions)")
                return True
            logger.info("Tool descriptions changed, recomputing embeddings")
        except (FileNotFoundError, KeyError):
            logger.info("No embedding cache found, computing from scratch")
        return False

    def _compute_embeddings(self):
        """Compute embeddings for all tool descriptions and save to disk cache."""
        # Validate threshold coverage
        missing_thresholds = set(TOOL_DESCRIPTIONS.keys()) - set(TOOL_THRESHOLDS.keys())
        if missing_thresholds:
            logger.warning(f"Tools without TOOL_THRESHOLDS entry (using default 0.45): {missing_thresholds}")

        # Batch tokenize all descriptions at once
        all_names = []
        all_texts = []
        for tool_name, descriptions in TOOL_DESCRIPTIONS.items():
            for desc in descriptions:
                all_texts.append(desc)
                all_names.append(tool_name)

        tokenized = self._tokenizer(
            all_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="np",
        )
        all_input_ids = tokenized["input_ids"].astype(np.int64)
        all_attention_masks = tokenized["attention_mask"].astype(np.int64)

        # Run HEF inference per description (batch=1 HEF constraint)
        all_embeddings = []
        for i in range(len(all_texts)):
            input_ids = all_input_ids[i:i+1]
            attention_mask = all_attention_masks[i:i+1]

            input_embeddings = self._text_embeddings[input_ids].astype(np.float32)
            attn_mask = _hef_attention_mask(attention_mask)

            self._bindings.input(EMBEDDING_INPUT_NAME).set_buffer(input_embeddings)
            self._bindings.input(MASK_INPUT_NAME).set_buffer(attn_mask)
            self._configured_model.run_async([self._bindings], lambda *_, **__: None).wait(TIMEOUT_MS)

            mask = attention_mask[:, :, None].astype(np.float32)
            sum_embeddings = np.sum(self._output_buffer * mask, axis=1)
            sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            all_embeddings.append(pooled / np.clip(norm, a_min=1e-9, a_max=None))

        self._tool_embeddings = np.vstack(all_embeddings)
        self._tool_names = all_names

        # Save cache to disk
        np.savez(
            TOOL_EMBEDDINGS_CACHE_PATH,
            embeddings=self._tool_embeddings,
            names=np.array(self._tool_names),
            hash=_descriptions_hash(),
        )
        logger.info(f"Computed and cached {len(all_names)} tool embeddings")

    def run(self, text: str) -> str:
        """Classify user text and return the best matching tool name."""
        if self._tool_embeddings is None or self._configured_model is None:
            raise RuntimeError("ToolSelector not initialized. Use as context manager or call __enter__() first.")

        if not text or not text.strip():
            logger.warning("Tool selector received empty input")
            return DEFAULT_TOOL

        user_embedding = self._embed_text(text)  # (1, hidden_dim)

        # Cosine similarity (embeddings are L2-normalized)
        similarities = np.dot(self._tool_embeddings, user_embedding.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_tool = self._tool_names[best_idx]
        best_score = float(similarities[best_idx])

        threshold = TOOL_THRESHOLDS.get(best_tool, 0.45)
        if best_score < threshold:
            best_tool = DEFAULT_TOOL

        logger.info(f"Tool selector: '{best_tool}' (confidence: {best_score:.3f})")
        return best_tool

    def close(self):
        if self._configured_model:
            del self._configured_model
            self._configured_model = None
        self._bindings = None
        self._output_buffer = None
        self._tool_embeddings = None
        self._tool_names = None
        self._text_embeddings = None
        if self._vdevice:
            self._vdevice = None

    def __enter__(self):
        self._configured_model = self._infer_model.configure()
        self._output_buffer = np.empty(self._infer_model.output(OUTPUT_NAME).shape, dtype=np.float32)
        self._bindings = self._configured_model.create_bindings()
        self._bindings.output(OUTPUT_NAME).set_buffer(self._output_buffer)
        if not self._load_embeddings_from_cache():
            self._compute_embeddings()
        logger.info("ToolSelector ready")
        return self

    def __exit__(self, *_):
        self.close()
