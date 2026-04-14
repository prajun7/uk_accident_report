from __future__ import annotations

import math
import os
from typing import Any

import joblib
import pandas as pd

RUSH_HOURS = {7, 8, 9, 17, 18, 19}

# Friendly defaults for partial payloads. Any missing fields are filled from this
# baseline and then the derived Step 6 features are recomputed.
BASELINE_DEFAULTS: dict[str, Any] = {
    "Speed_limit": 30,
    "Road_Type": 3,
    "1st_Road_Class": 3,
    "1st_Road_Number": 0.0,
    "2nd_Road_Class": 6.0,
    "2nd_Road_Number": 0.0,
    "Junction_Detail": 0.0,
    "Junction_Control": 4.0,
    "Light_Conditions": 1,
    "Weather_Conditions": 1.0,
    "Road_Surface_Conditions": 1.0,
    "Special_Conditions_at_Site": 0.0,
    "Carriageway_Hazards": 0.0,
    "Pedestrian_Crossing-Human_Control": 0.0,
    "Pedestrian_Crossing-Physical_Facilities": 0.0,
    "Day_of_Week": 6,
    "Urban_or_Rural_Area": 1,
    "Hour": 14,
    "IsNight": 0,
}

INTEGER_LIKE_COLUMNS = {
    "Speed_limit",
    "Road_Type",
    "1st_Road_Class",
    "Light_Conditions",
    "Day_of_Week",
    "Urban_or_Rural_Area",
    "Hour",
    "IsNight",
}


class RiskZonePredictor:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "output")
        self.bundle_path = os.path.join(self.output_dir, "7_model_bundle.joblib")
        self.features_path = os.path.join(self.output_dir, "6_X_final.csv")

        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(
                f"Missing trained model bundle at {self.bundle_path}. Re-run Step 7 first."
            )

        self.bundle = joblib.load(self.bundle_path)
        self.model = self.bundle["model"]
        self.model_name = self.bundle["model_name"]
        self.target_names = self.bundle.get(
            "target_names", ["Low Risk", "Medium Risk", "High Risk"]
        )
        self.raw_feature_columns = list(self.bundle["raw_feature_columns"])
        self.feature_columns = list(self.bundle["feature_columns"])
        self.target_encoder = self.bundle.get("target_encoder")
        self.target_encoder_cols = list(self.bundle.get("target_encoder_cols", []))
        self.nn_scaler = self.bundle.get("nn_scaler")

        self.class_ids = [
            int(class_id)
            for class_id in getattr(self.model, "classes_", range(len(self.target_names)))
        ]
        self.class_to_label = dict(zip(self.class_ids, self.target_names))
        self.default_payload = self._load_default_payload()

    def _load_default_payload(self) -> dict[str, Any]:
        defaults = dict(BASELINE_DEFAULTS)

        if os.path.exists(self.features_path):
            features_df = pd.read_csv(self.features_path)
            for col in self.raw_feature_columns:
                if col in {"Hour_Sin", "Hour_Cos", "IsRushHour", "DayNight_Context", "Urban_Speed_Net", "Junction_Complexity"}:
                    continue
                if col not in features_df.columns:
                    continue
                series = features_df[col].dropna()
                if series.empty:
                    continue
                defaults[col] = series.mode().iloc[0]

        defaults.update(BASELINE_DEFAULTS)
        return self._apply_derived_features(defaults, force_auto_isnight=True)

    def _coerce_value(self, col: str, value: Any) -> Any:
        if value is None or value == "":
            return self.default_payload.get(col)

        if col == "DayNight_Context":
            return str(value)

        if col in INTEGER_LIKE_COLUMNS:
            return int(round(float(value)))

        return float(value)

    def _apply_derived_features(
        self, payload: dict[str, Any], force_auto_isnight: bool = False
    ) -> dict[str, Any]:
        row = dict(payload)
        fallback_defaults = getattr(self, "default_payload", BASELINE_DEFAULTS)

        hour = int(round(float(row.get("Hour", fallback_defaults.get("Hour", 14)))))
        hour = max(0, min(23, hour))
        row["Hour"] = hour

        if force_auto_isnight:
            isnight = int(hour < 6 or hour >= 20)
        else:
            isnight = int(round(float(row.get("IsNight", int(hour < 6 or hour >= 20)))))
        row["IsNight"] = isnight

        day_of_week = int(
            round(float(row.get("Day_of_Week", fallback_defaults.get("Day_of_Week", 6))))
        )
        day_of_week = max(1, min(7, day_of_week))
        row["Day_of_Week"] = day_of_week

        speed_limit = int(
            round(float(row.get("Speed_limit", fallback_defaults.get("Speed_limit", 30))))
        )
        row["Speed_limit"] = max(0, speed_limit)

        urban_or_rural = int(
            round(
                float(
                    row.get(
                        "Urban_or_Rural_Area",
                        fallback_defaults.get("Urban_or_Rural_Area", 1),
                    )
                )
            )
        )
        urban_or_rural = max(1, min(3, urban_or_rural))
        row["Urban_or_Rural_Area"] = urban_or_rural

        junction_detail = float(
            row.get("Junction_Detail", fallback_defaults.get("Junction_Detail", 0.0))
        )
        junction_control = float(
            row.get("Junction_Control", fallback_defaults.get("Junction_Control", 4.0))
        )
        row["Junction_Detail"] = junction_detail
        row["Junction_Control"] = junction_control

        row["IsRushHour"] = int(hour in RUSH_HOURS)
        row["Hour_Sin"] = math.sin(2 * math.pi * hour / 24.0)
        row["Hour_Cos"] = math.cos(2 * math.pi * hour / 24.0)
        row["DayNight_Context"] = f"{day_of_week}_{isnight}"
        row["Urban_Speed_Net"] = urban_or_rural * row["Speed_limit"]
        row["Junction_Complexity"] = junction_detail + junction_control

        return row

    def prepare_payload(self, payload: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
        payload = payload or {}
        raw_row = dict(self.default_payload)
        ignored_fields: list[str] = []
        defaults_applied: list[str] = []

        for key, value in payload.items():
            if key in self.raw_feature_columns or key == "Time":
                continue
            ignored_fields.append(key)

        if "Time" in payload and "Hour" not in payload:
            time_value = str(payload["Time"]).strip()
            hour_text = time_value.split(":", 1)[0]
            if hour_text:
                raw_row["Hour"] = self._coerce_value("Hour", hour_text)

        manual_isnight = "IsNight" in payload and payload["IsNight"] not in (None, "")

        for col in self.raw_feature_columns:
            if col in {"IsRushHour", "Hour_Sin", "Hour_Cos", "DayNight_Context", "Urban_Speed_Net", "Junction_Complexity"}:
                continue
            if col in payload:
                raw_row[col] = self._coerce_value(col, payload[col])
            else:
                defaults_applied.append(col)

        raw_row = self._apply_derived_features(raw_row, force_auto_isnight=not manual_isnight)
        frame = pd.DataFrame([{col: raw_row[col] for col in self.raw_feature_columns}])
        return frame, defaults_applied, ignored_fields

    def transform(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        transformed = raw_frame.copy()

        if self.target_encoder is not None and self.target_encoder_cols:
            for col in self.target_encoder_cols:
                if col in transformed.columns:
                    transformed[col] = transformed[col].astype("string")
            transformed = self.target_encoder.transform(transformed)

        return transformed.reindex(columns=self.feature_columns, fill_value=0)

    def predict(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw_frame, defaults_applied, ignored_fields = self.prepare_payload(payload)
        model_input = self.transform(raw_frame)
        predict_input = (
            self.nn_scaler.transform(model_input)
            if self.nn_scaler is not None
            else model_input
        )

        prediction = int(self.model.predict(predict_input)[0])
        label = self.class_to_label.get(prediction, str(prediction))

        out: dict[str, Any] = {
            "label": label,
            "prediction": prediction,
            "model_name": self.model_name,
            "defaults_applied": defaults_applied,
            "ignored_fields": ignored_fields,
            "raw_features_used": raw_frame.iloc[0].to_dict(),
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(predict_input)[0]
            prob_map = {
                self.class_to_label.get(class_id, str(class_id)): float(prob)
                for class_id, prob in zip(self.class_ids, probs)
            }
            out["probabilities"] = prob_map
            out["detail"] = ", ".join(
                f"{name}={prob_map[name]:.3f}" for name in self.target_names if name in prob_map
            )
            if "High Risk" in prob_map:
                out["high_risk_probability"] = prob_map["High Risk"]

        return out

    def schema(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "bundle_path": self.bundle_path,
            "raw_feature_columns": self.raw_feature_columns,
            "model_feature_columns": self.feature_columns,
            "target_encoder_columns": self.target_encoder_cols,
            "defaults": self.default_payload,
            "labels": self.target_names,
            "derived_features": [
                "IsNight",
                "IsRushHour",
                "Hour_Sin",
                "Hour_Cos",
                "DayNight_Context",
                "Urban_Speed_Net",
                "Junction_Complexity",
            ],
            "accepted_optional_fields": ["Time"],
        }
