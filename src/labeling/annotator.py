"""
Data Annotator for NILM AI
Orchestrates multi-tier label generation: Binary ON/OFF, Multi-State Class ID,
Continuous Regression Target Power, and Transition Events.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from .state_classifier import StateClassifier, TransitionEvent
from .state_definitions import get_appliance_config


class DataAnnotator:
    """Generates 4-tier NILM annotations for preprocessed electrical datasets."""

    def __init__(self, sampling_hz: float = 60.0):
        self.sampling_hz = sampling_hz

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        appliance_type: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[TransitionEvent], Dict]:
        """Annotates a preprocessed DataFrame with 4-Tier ground truth labels.

        Returns:
            annotated_df: DataFrame with added label columns
            events: List of detected TransitionEvents
            summary: Statistical summary of states and events
        """
        if appliance_type is None:
            appliance_type = df.get("appliance_type", pd.Series(["unknown"])).iloc[0]

        config = get_appliance_config(appliance_type)
        classifier = StateClassifier(config=config, sampling_hz=self.sampling_hz)

        p_vals = df["p_w"].values if "p_w" in df.columns else df["p_target_w"].values
        q_vals = df["q_var"].values if "q_var" in df.columns else None
        t_vals = df["t_rel_s"].values if "t_rel_s" in df.columns else None

        state_ids, is_on, events = classifier.classify_series(p_vals, q_vals, t_vals)

        state_names = [classifier.state_map.get(s_id, f"STATE_{s_id}") for s_id in state_ids]

        out = df.copy()
        out["is_on"] = is_on
        out["state_id"] = state_ids
        out["state_name"] = state_names

        # Continuous regression ground truth target power (P_target)
        # When device is OFF, ground truth is 0.0W. When ON, it is the clean active power.
        clean_p = out["p_target_w"].values if "p_target_w" in out.columns else np.maximum(0.0, p_vals)
        out["target_power_w"] = np.where(is_on == 1, clean_p, 0.0)

        # Compute State Statistics
        total_samples = len(out)
        state_distribution = {}
        for state in config.states:
            count = int((state_ids == state.state_id).sum())
            pct = round(count / total_samples * 100.0, 2) if total_samples > 0 else 0.0
            duration_min = round(count / self.sampling_hz / 60.0, 2)
            state_distribution[state.name] = {
                "state_id": state.state_id,
                "description": state.description,
                "count": count,
                "percentage": pct,
                "duration_min": duration_min,
            }

        event_counts = {
            "total_events": len(events),
            "on_events": sum(1 for e in events if e.event_type == "ON"),
            "off_events": sum(1 for e in events if e.event_type == "OFF"),
            "mode_change_events": sum(1 for e in events if e.event_type == "MODE_CHANGE"),
        }

        summary = {
            "appliance_type": appliance_type,
            "korean_name": config.korean_name,
            "total_samples": total_samples,
            "duration_s": round(total_samples / self.sampling_hz, 2),
            "on_percentage": round(float(is_on.mean()) * 100.0, 2),
            "state_distribution": state_distribution,
            "events_summary": event_counts,
        }

        return out, events, summary

    def save_annotations(
        self,
        df_annotated: pd.DataFrame,
        events: List[TransitionEvent],
        summary: Dict,
        output_prefix: Union[str, Path],
    ) -> Dict[str, str]:
        """Saves annotated CSV, events JSON, and summary JSON."""
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        csv_path = prefix.with_suffix(".csv")
        events_path = prefix.with_name(f"{prefix.stem}_events.json")
        summary_path = prefix.with_name(f"{prefix.stem}_summary.json")

        df_annotated.to_csv(csv_path, index=False)

        with open(events_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in events], f, indent=2, ensure_ascii=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return {
            "dataset_csv": str(csv_path),
            "events_json": str(events_path),
            "summary_json": str(summary_path),
        }
