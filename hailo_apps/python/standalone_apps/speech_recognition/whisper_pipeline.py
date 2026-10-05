"""
Whisper inference pipeline for Hailo-8/8L/10H.

Uses the low-level HailoRT InferModel API with separate encoder + decoder HEFs.
Runs inference in a background thread; feed mel spectrograms via send_data(),
retrieve transcriptions via get_transcription().
"""

import logging
import os
import numpy as np
from queue import Queue, Empty
from threading import Thread

from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm, FormatType
from transformers import AutoTokenizer

try:
    from .postprocessing import apply_repetition_penalty
except ImportError:
    from postprocessing import apply_repetition_penalty

logger = logging.getLogger(__name__)


class WhisperPipeline:
    """Encoder–decoder Whisper pipeline running on any Hailo accelerator."""

    def __init__(self, encoder_path: str, decoder_path: str,
                 variant: str = "base", npy_dir: str = None,
                 add_embed: bool = False):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.variant = variant
        self.add_embed = add_embed
        self.timeout_ms = 100_000_000

        # Load decoder tokenization assets from central resources/npy/ directory
        if npy_dir is None:
            npy_dir = "/usr/local/hailo/resources/npy"
        self._npy_dir = npy_dir
        self.token_embedding_weight = np.load(
            os.path.join(self._npy_dir,
                         f"token_embedding_weight_{variant}.npy")
        )
        self.onnx_add_input = np.load(
            os.path.join(self._npy_dir,
                         f"onnx_add_input_{variant}.npy")
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{variant}")

        # Forced decoder prefix, derived from the actual tokenizer in use.
        # Multilingual checkpoints (base, small, medium, large, ...) expect
        # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>.
        # English-only checkpoints (tiny.en, base.en, small.en, medium.en) have
        # no language/task tokens in their vocabulary at all, so forcing the
        # multilingual prefix there produces garbage text. Their prefix is just
        # <|startoftranscript|> <|notimestamps|>.
        self.forced_decoder_ids = self._build_forced_decoder_ids()

        # Query encoder input shape to determine chunk length in seconds
        encoder_hef = HEF(self.encoder_path)
        self.input_audio_length = int(
            encoder_hef.get_input_vstream_infos()[0].shape[1] / 100
        )

        self.decoding_sequence_length = None  # set from decoder HEF
        self._data_q = Queue()
        self._results_q = Queue()
        self._running = True
        self._thread = Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _build_forced_decoder_ids(self):
        """Build the forced decoder prefix, tokenizer/variant aware (see __init__)."""
        bos = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        notimestamps = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        if self.variant.endswith(".en"):
            return [bos, notimestamps]

        lang = self.tokenizer.convert_tokens_to_ids("<|en|>")
        task = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        return [bos, lang, task, notimestamps]

    def _tokenization(self, decoder_input_ids):
        """Manual token embedding lookup (replaces embedding layer on host).

        Hailo-8/8L decoders were compiled without the Add/Gather tokenization
        operators, so Add + Unsqueeze + Transpose must run on host.
        Hailo-10H decoders include the Add in the HEF.
        """
        gather = self.token_embedding_weight[decoder_input_ids]
        if self.add_embed:
            # H8/H8L: Add + Unsqueeze + Transpose on host
            add_output = gather + self.onnx_add_input
            unsqueeze_output = np.expand_dims(add_output, axis=0)
            return np.transpose(unsqueeze_output, (0, 2, 1, 3))
        # H10H: only Gather + expand
        return np.expand_dims(gather, axis=0)

    def _inference_loop(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"

        decoder_hef = HEF(self.decoder_path)
        sorted_output_names = decoder_hef.get_sorted_output_names()
        decoder_model_name = decoder_hef.get_network_group_names()[0]
        self.decoding_sequence_length = (
            decoder_hef.get_output_vstream_infos()[0].shape[1]
        )

        useful_outputs = [n for n in sorted_output_names if "conv" in n]

        with VDevice(params) as vdevice:
            enc_model = vdevice.create_infer_model(self.encoder_path)
            dec_model = vdevice.create_infer_model(self.decoder_path)

            enc_model.input().set_format_type(FormatType.FLOAT32)
            enc_model.output().set_format_type(FormatType.FLOAT32)
            dec_model.input(f"{decoder_model_name}/input_layer1").set_format_type(
                FormatType.FLOAT32)
            dec_model.input(f"{decoder_model_name}/input_layer2").set_format_type(
                FormatType.FLOAT32)
            for name in sorted_output_names:
                dec_model.output(name).set_format_type(FormatType.FLOAT32)

            with enc_model.configure() as enc_cfg, dec_model.configure() as dec_cfg:
                enc_bindings = enc_cfg.create_bindings()
                dec_bindings = dec_cfg.create_bindings()

                while self._running:
                    try:
                        input_mel = self._data_q.get(timeout=1)
                    except Empty:
                        continue

                    # --- Encoder ---
                    input_mel = np.ascontiguousarray(input_mel)
                    enc_bindings.input().set_buffer(input_mel)
                    enc_buf = np.zeros(enc_model.output().shape, dtype=np.float32)
                    enc_bindings.output().set_buffer(enc_buf)
                    enc_cfg.run([enc_bindings], self.timeout_ms)
                    encoded = enc_bindings.output().get_buffer()

                    logger.debug(
                        "Encoder: in=%s (%.3f–%.3f), out=%s (%.3f–%.3f)",
                        input_mel.shape, input_mel.min(), input_mel.max(),
                        encoded.shape, encoded.min(), encoded.max(),
                    )

                    # --- Decoder (autoregressive) ---
                    seq_len = self.decoding_sequence_length
                    dec_ids = np.zeros((1, seq_len), dtype=np.int64)
                    for k, tok in enumerate(self.forced_decoder_ids):
                        dec_ids[0][k] = tok
                    # Free generation starts after the forced prefix
                    free_start = len(self.forced_decoder_ids) - 1
                    generated = []

                    for i in range(free_start, seq_len - 1):
                        tok_embed = self._tokenization(dec_ids)

                        dec_bindings.input(
                            f"{decoder_model_name}/input_layer1").set_buffer(encoded)
                        dec_bindings.input(
                            f"{decoder_model_name}/input_layer2").set_buffer(tok_embed)

                        for name in sorted_output_names:
                            buf = np.zeros(
                                dec_model.output(name).shape, dtype=np.float32)
                            dec_bindings.output(name).set_buffer(buf)

                        dec_cfg.run([dec_bindings], self.timeout_ms)

                        outputs = np.concatenate(
                            [dec_bindings.output(n).get_buffer()
                             for n in useful_outputs],
                            axis=2,
                        )

                        logits = apply_repetition_penalty(
                            outputs[:, i], generated, penalty=1.5)
                        next_token = int(np.argmax(logits))
                        generated.append(next_token)
                        dec_ids[0][i + 1] = next_token

                        if next_token == self.tokenizer.eos_token_id:
                            break

                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    logger.debug("Decoded %d tokens: %r", len(generated), text)
                    self._results_q.put(text)

    def get_chunk_length(self) -> int:
        """Expected audio chunk length in seconds."""
        return self.input_audio_length

    def send_data(self, mel_spectrogram: np.ndarray):
        """Feed a preprocessed mel spectrogram chunk."""
        self._data_q.put(mel_spectrogram)

    def get_transcription(self) -> str:
        """Block until next transcription result is available."""
        return self._results_q.get()

    def stop(self):
        """Signal the inference thread to exit and wait."""
        self._running = False
        self._thread.join(timeout=5)
