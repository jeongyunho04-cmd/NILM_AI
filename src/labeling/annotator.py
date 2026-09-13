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
        # 타임라인 이어붙인 자리에서 나온 전이는 실제 사건이 아닐 수 있으므로 표시해 둔다.
        seam_vals = df["is_segment_seam"].values if "is_segment_seam" in df.columns else None

        state_ids, is_on, events = classifier.classify_series(p_vals, q_vals, t_vals, seam_vals)
        state_ids, is_on, events = self._apply_armed_state(
            config, classifier, df, state_ids, is_on, events, t_vals)

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
            # 이어붙인 경계에서 난 전이는 실제 기기 동작이 아닐 수 있다.
            "seam_suspect_events": sum(1 for e in events if e.at_segment_seam),
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

    def _apply_armed_state(self, config, classifier, df, state_ids, is_on, events, t_vals):
        """바닥 준위로 '플러그만'(0) 과 '스위치 켜짐·릴레이 열림'(armed) 을 가른다 (규칙 5).

        통전 상태(`on_state_min_id` 이상)는 건드리지 않는다. 나머지 사이클은 `p_target_w`(파일 바닥을 뺀 전력)의
        `armed_window_s` 구름 10백분위 — 펄스가 섞여도 바닥을 가리킨다 — 가 `armed_floor_w` 이상이면 armed,
        아니면 0. 원시 p_w 로는 못 가른다: 계측계 바닥이 파일마다 1.33~1.60W 로 달라 절대 문턱이 없다.
        """
        armed = getattr(config, "armed_state_id", None)
        if armed is None or "p_target_w" not in df.columns or len(state_ids) == 0:
            return state_ids, is_on, events
        on_min = getattr(config, "on_state_min_id", None)
        thr_on = int(on_min) if on_min is not None else 1
        win = max(3, int(round(self.sampling_hz * float(getattr(config, "armed_window_s", 4.0)))))
        base = pd.Series(df["p_target_w"].values.astype(float)).rolling(
            window=win, center=True, min_periods=1).quantile(0.10).values
        not_heating = state_ids < thr_on
        new_ids = state_ids.copy()
        new_ids[not_heating & (base >= float(getattr(config, "armed_floor_w", 0.3)))] = int(armed)
        new_ids[not_heating & (base < float(getattr(config, "armed_floor_w", 0.3)))] = 0
        # 0 <-> armed 전이를 사건으로 남긴다 (분류기는 원시 p_w 만 봐서 이 전이를 모른다)
        p_w = df["p_w"].values.astype(float) if "p_w" in df.columns else df["p_target_w"].values.astype(float)
        w = int(self.sampling_hz * 0.5)
        low = (new_ids < thr_on)
        ch = np.flatnonzero(np.diff(new_ids) != 0) + 1
        for k in ch:
            a, b = int(new_ids[k - 1]), int(new_ids[k])
            if not (low[k - 1] and low[k]):
                continue                       # 통전 관련 전이는 분류기가 이미 적었다
            pb = float(np.median(p_w[max(0, k - w):k])); pa = float(np.median(p_w[k:min(len(p_w), k + w)]))
            events.append(TransitionEvent(
                sample_idx=int(k), t_s=round(float(t_vals[k]) if t_vals is not None else k / self.sampling_hz, 3),
                from_state_id=a, from_state_name=classifier.state_map.get(a, f"STATE_{a}"),
                to_state_id=b, to_state_name=classifier.state_map.get(b, f"STATE_{b}"),
                event_type=("ON" if a == 0 else "OFF" if b == 0 else "MODE_CHANGE"),
                p_before_w=round(pb, 2), p_after_w=round(pa, 2), delta_p_w=round(pa - pb, 2), delta_q_var=0.0,
                prev_state_duration_s=0.0, at_segment_seam=False))
        events.sort(key=lambda e: e.sample_idx)
        new_on = np.where(new_ids >= thr_on, 1, 0).astype(int)
        return new_ids, new_on, events

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
