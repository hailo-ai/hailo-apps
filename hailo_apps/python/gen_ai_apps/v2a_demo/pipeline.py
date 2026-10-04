"""Voice-to-Action pipeline — orchestrates all five processing stages.

Stages: STT -> Tool Selection -> LLM Parameter Extraction -> Tool Execution -> TTS
"""

import logging
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional

from hailo_platform import VDevice
from stt import STTEngine
from tool_selector import ToolSelector
from llm import LLMEngine
from tts import TTSEngine
from tools import run_tool
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID

logger = logging.getLogger("v2a_demo")

PIPELINE_STAGE_LABELS = {
    "stt": "Speech2Text",
    "tool_select": "Tool Selection",
    "llm": "LLM Processing",
    "tool": "Tool Execution",
    "tts": "TTS Synthesis",
}


class V2APipeline:
    """Voice-to-Action Pipeline: Speech2Text -> ToolSelector -> LLM -> Tool -> Text2Speech"""

    def __init__(self, tts_output_path: Optional[str] = None):
        logger.info("Initializing pipeline components...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self.vdevice = VDevice(params)
        self.tts_output_path = tts_output_path
        self.stt = STTEngine(self.vdevice)
        self.tool_selector = ToolSelector(self.vdevice)
        self.llm = LLMEngine(self.vdevice)
        self.tts = TTSEngine()
        logger.info("Pipeline initialized successfully.")

    def process_audio(self, audio_data: np.ndarray) -> str:
        timing = {}

        # Stage 1: Speech-to-Text
        t0 = time.perf_counter()
        text = self.stt.run(audio_data)
        timing["stt"] = time.perf_counter() - t0
        logger.info(f"Transcription: {text}")

        # Stage 2: Tool Selection
        t0 = time.perf_counter()
        tool_name = self.tool_selector.run(text)
        timing["tool_select"] = time.perf_counter() - t0
        logger.info(f"Selected tool: {tool_name}")

        # Stage 3: Parameter Extraction (LLM)
        t0 = time.perf_counter()
        params = self.llm.run(text, tool_name)
        timing["llm"] = time.perf_counter() - t0
        logger.debug(f"Extracted params: {params}")

        # Stage 4: Tool Execution
        t0 = time.perf_counter()
        tool_response = run_tool(tool_name, params)
        timing["tool"] = time.perf_counter() - t0
        logger.info(f"Tool response: {tool_response}")

        # Stage 5: Text-to-Speech
        logger.info("Generating speech...")
        t0 = time.perf_counter()
        tts_result = self.tts.run(tool_response)
        timing["tts"] = time.perf_counter() - t0

        if tts_result:
            audio_array, sample_rate = tts_result
            if self.tts_output_path:
                sf.write(self.tts_output_path, audio_array, sample_rate)
            else:
                sd.play(audio_array, sample_rate)
                sd.wait()

        self._log_timing(timing)
        return tool_response

    def _log_timing(self, timing: dict):
        logger.info("---- Pipeline Performance ----")
        for key in ("stt", "tool_select", "llm", "tool", "tts"):
            if key in timing:
                logger.info(f"{PIPELINE_STAGE_LABELS[key]:<17}: {timing[key]:>6.3f} s")

        total = sum(timing.values())
        logger.info(f"{'Total':<17}: {total:>6.3f} s")
        logger.info("------------------------------")

    def close(self):
        self.stt.close()
        self.tool_selector.close()
        self.llm.close()
        self.tts.close()
        self.vdevice.release()

    def __enter__(self):
        self.vdevice.__enter__()
        self.stt.__enter__()
        self.tool_selector.__enter__()
        self.llm.__enter__()
        self.tts.__enter__()
        return self

    def __exit__(self, *_):
        self.close()
