"""
State Manager for LLM context states.

Manages saving, loading, and listing context states for tools.
Each state includes:
- Binary LLM context (.state)
- YAML config snapshot (.yaml)
- Metadata and metrics (.meta.json)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hailo_platform.genai import LLM

logger = logging.getLogger(__name__)

# Try to import yaml for config snapshots
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class StateMetrics:
    """
    Performance metrics for a saved state.

    Attributes:
        tool_call_accuracy: Percentage of queries where correct tool was called.
        e2e_accuracy: End-to-end accuracy percentage.
        avg_latency_ms: Average response latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        no_tool_precision: Precision for correctly NOT calling tools.
        test_cases_passed: Number of test cases passed.
        test_cases_total: Total number of test cases.
    """

    tool_call_accuracy: float = 0.0
    e2e_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    no_tool_precision: float = 0.0
    test_cases_passed: int = 0
    test_cases_total: int = 0


@dataclass
class StateInfo:
    """
    Information about a saved state.

    Attributes:
        state_name: Name of the state.
        created: Creation timestamp.
        yaml_hash: SHA256 hash of the YAML config.
        context_tokens: Number of tokens in context.
        few_shot_count: Number of few-shot examples.
        parent_state: Name of parent state (if derived).
        performance: Performance metrics.
        notes: Optional notes about this state.
    """

    state_name: str
    created: str = ""
    yaml_hash: str = ""
    context_tokens: int = 0
    few_shot_count: int = 0
    parent_state: str = ""
    performance: StateMetrics = field(default_factory=StateMetrics)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "state_name": self.state_name,
            "created": self.created,
            "yaml_hash": self.yaml_hash,
            "context_tokens": self.context_tokens,
            "few_shot_count": self.few_shot_count,
            "parent_state": self.parent_state,
            "performance": asdict(self.performance),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateInfo":
        """
        Create from dictionary.

        Args:
            data: Dictionary data.

        Returns:
            StateInfo instance.
        """
        perf_data = data.get("performance", {})
        return cls(
            state_name=data.get("state_name", ""),
            created=data.get("created", ""),
            yaml_hash=data.get("yaml_hash", ""),
            context_tokens=data.get("context_tokens", 0),
            few_shot_count=data.get("few_shot_count", 0),
            parent_state=data.get("parent_state", ""),
            performance=StateMetrics(**perf_data) if perf_data else StateMetrics(),
            notes=data.get("notes", ""),
        )


class StateManager:
    """
    Manages context states for a tool.

    States are stored in the tool's contexts/ directory:
    - {state_name}.state - Binary LLM context
    - {state_name}.yaml - YAML config snapshot
    - {state_name}.meta.json - Metadata and metrics
    """

    def __init__(self, tool_name: str, contexts_dir: Optional[Path] = None):
        """
        Initialize state manager for a tool.

        Args:
            tool_name: Name of the tool.
            contexts_dir: Optional custom directory for contexts.
                         Defaults to tools/{tool_name}/contexts/
        """
        self.tool_name = tool_name
        if contexts_dir is not None:
            self.contexts_dir = contexts_dir
        else:
            # Default: relative to this module's parent
            base_dir = Path(__file__).parent / "tools" / tool_name / "contexts"
            self.contexts_dir = base_dir

        self._current_state_name: Optional[str] = None

    def _ensure_dir(self) -> bool:
        """
        Ensure contexts directory exists.

        Returns:
            True if directory exists or was created.
        """
        try:
            self.contexts_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error("Failed to create contexts dir: %s", e)
            return False

    def _get_state_path(self, state_name: str) -> Path:
        """Get path to state file."""
        return self.contexts_dir / f"{state_name}.state"

    def _get_yaml_path(self, state_name: str) -> Path:
        """Get path to YAML snapshot file."""
        return self.contexts_dir / f"{state_name}.yaml"

    def _get_meta_path(self, state_name: str) -> Path:
        """Get path to metadata file."""
        return self.contexts_dir / f"{state_name}.meta.json"

    def _compute_yaml_hash(self, yaml_config: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of YAML config.

        Args:
            yaml_config: Configuration dictionary.

        Returns:
            SHA256 hash string.
        """
        # Serialize deterministically
        config_str = json.dumps(yaml_config, sort_keys=True, ensure_ascii=False)
        return f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()[:16]}"

    def save_state(
        self,
        state_name: str,
        llm: "LLM",
        yaml_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[StateMetrics] = None,
        parent_state: str = "",
        notes: str = "",
    ) -> bool:
        """
        Save LLM context with YAML snapshot and optional metrics.

        Creates:
        - {state_name}.state - Binary LLM context
        - {state_name}.yaml - YAML config snapshot (for reproducibility)
        - {state_name}.meta.json - Metadata and performance metrics

        Args:
            state_name: Name for this state.
            llm: LLM instance with context to save.
            yaml_config: YAML configuration dict to snapshot.
            metrics: Optional performance metrics.
            parent_state: Name of parent state if this is derived.
            notes: Optional notes about this state.

        Returns:
            True if save was successful.
        """
        if not self._ensure_dir():
            return False

        state_path = self._get_state_path(state_name)
        yaml_path = self._get_yaml_path(state_name)
        meta_path = self._get_meta_path(state_name)

        try:
            # 1. Save binary context
            context_data = llm.save_context()
            if not context_data:
                logger.warning("LLM returned empty context, skipping save")
                return False

            # Atomic write for state file
            temp_state = state_path.with_suffix(".tmp")
            with open(temp_state, "wb") as f:
                f.write(context_data)
            shutil.move(str(temp_state), str(state_path))

            # 2. Save YAML snapshot
            yaml_hash = ""
            few_shot_count = 0
            if yaml_config and YAML_AVAILABLE:
                yaml_hash = self._compute_yaml_hash(yaml_config)
                few_shot_count = len(yaml_config.get("few_shot_examples", []))

                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)

            # 3. Save metadata and validate capacity
            try:
                context_tokens = llm.get_context_usage_size()
                max_capacity = llm.max_context_capacity()

                # Validate that context doesn't exceed capacity
                if context_tokens > max_capacity:
                    logger.error(
                        "Context exceeds capacity: %d tokens > %d max capacity. "
                        "State may not load correctly.",
                        context_tokens, max_capacity
                    )
                    # exit the program
                    sys.exit(1)
                elif context_tokens > max_capacity * 0.95:
                    logger.warning(
                        "Context near capacity: %d/%d tokens (%.1f%%)",
                        context_tokens, max_capacity, (context_tokens * 100.0 / max_capacity)
                    )
            except Exception as e:
                logger.warning("Could not validate context capacity: %s", e)
                context_tokens = 0
                max_capacity = 0

            state_info = StateInfo(
                state_name=state_name,
                created=datetime.utcnow().isoformat() + "Z",
                yaml_hash=yaml_hash,
                context_tokens=context_tokens,
                few_shot_count=few_shot_count,
                parent_state=parent_state,
                performance=metrics or StateMetrics(),
                notes=notes,
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(state_info.to_dict(), f, indent=2)

            self._current_state_name = state_name
            logger.info("Saved state: %s (tokens: %d)", state_name, context_tokens)
            return True

        except Exception as e:
            logger.error("Failed to save state %s: %s", state_name, e)
            # Cleanup temp files
            for temp in [state_path.with_suffix(".tmp")]:
                try:
                    if temp.exists():
                        temp.unlink()
                except Exception:
                    pass
            return False

    def load_state(self, state_name: str, llm: "LLM") -> bool:
        """
        Load a saved context state into the LLM.

        Args:
            state_name: Name of the state to load.
            llm: LLM instance to load context into.

        Returns:
            True if load was successful.
        """
        state_path = self._get_state_path(state_name)

        if not state_path.exists():
            logger.debug("State not found: %s", state_name)
            return False

        try:
            with open(state_path, "rb") as f:
                context_data = f.read()

            if not context_data:
                logger.warning("State file is empty: %s", state_name)
                return False

            llm.load_context(context_data)
            self._current_state_name = state_name

            # Verify load was successful
            try:
                loaded_tokens = llm.get_context_usage_size()
                logger.debug("Loaded state: %s (%d tokens)", state_name, loaded_tokens)
            except Exception:
                logger.debug("Loaded state: %s", state_name)

            return True

        except Exception as e:
            logger.error("Failed to load state %s: %s", state_name, e)
            return False

    def reload_state(self, llm: "LLM") -> bool:
        """
        Reload the current state (useful for context reset).

        Args:
            llm: LLM instance.

        Returns:
            True if reload was successful.
        """
        if not self._current_state_name:
            logger.warning("No current state to reload")
            return False

        return self.load_state(self._current_state_name, llm)

    def list_states(self) -> List[StateInfo]:
        """
        List all saved states with their metrics.

        Returns:
            List of StateInfo objects.
        """
        states: List[StateInfo] = []

        if not self.contexts_dir.exists():
            return states

        for meta_path in self.contexts_dir.glob("*.meta.json"):
            state_name = meta_path.stem.replace(".meta", "")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                states.append(StateInfo.from_dict(data))
            except Exception as e:
                logger.warning("Failed to read metadata for %s: %s", state_name, e)
                # Add minimal info
                states.append(StateInfo(state_name=state_name))

        states.sort(key=lambda s: s.created, reverse=True)
        return states

    def get_best_state(self, metric: str = "tool_call_accuracy") -> Optional[str]:
        """
        Get the state name with best performance on given metric.

        Args:
            metric: Metric name to compare (default: tool_call_accuracy).

        Returns:
            State name with best score, or None if no states exist.
        """
        states = self.list_states()
        if not states:
            return None

        def get_metric_value(state: StateInfo) -> float:
            return getattr(state.performance, metric, 0.0)

        best = max(states, key=get_metric_value)
        return best.state_name

    def delete_state(self, state_name: str) -> bool:
        """
        Delete a saved state.

        Args:
            state_name: Name of the state to delete.

        Returns:
            True if deletion was successful.
        """
        deleted_any = False

        for path in [
            self._get_state_path(state_name),
            self._get_yaml_path(state_name),
            self._get_meta_path(state_name),
        ]:
            try:
                if path.exists():
                    path.unlink()
                    deleted_any = True
            except Exception as e:
                logger.warning("Failed to delete %s: %s", path, e)

        if deleted_any:
            logger.info("Deleted state: %s", state_name)

        return deleted_any

    def get_state_info(self, state_name: str) -> Optional[StateInfo]:
        """
        Get information about a specific state.

        Args:
            state_name: Name of the state.

        Returns:
            StateInfo if found, None otherwise.
        """
        meta_path = self._get_meta_path(state_name)

        if not meta_path.exists():
            # Check if state file exists at least
            if self._get_state_path(state_name).exists():
                return StateInfo(state_name=state_name)
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StateInfo.from_dict(data)
        except Exception as e:
            logger.warning("Failed to read state info for %s: %s", state_name, e)
            return None

    @property
    def current_state(self) -> Optional[str]:
        """Get the name of the currently loaded state."""
        return self._current_state_name

