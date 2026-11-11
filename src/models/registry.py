import os
import json
import random
import joblib
from typing import Dict, Any, List, Optional
from src.models.rf_model import RandomForestModel
from src.models.lgb_model import LightGBMModel
from src.models.xgb_model import XGBoostModel

class ModelRegistry:
    def __init__(self, bank_path: str = "models/multi_model_bank.joblib", meta_path: str = "models/multi_model_metadata.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.class_names = self.metadata["class_names"]
        self.feature_names = self.metadata["feature_names"]
        self.leaderboard = self.metadata["leaderboard"]
        self.mcnemar = self.metadata["mcnemar"]

        # Load precomputed cases
        self.cases: List[Dict[str, Any]] = joblib.load(bank_path)
        self.case_map = {c["case_id"]: c for c in self.cases}

        # Lazy models container
        self._models: Dict[str, Any] = {}

    def get_model(self, name: str):
        if name not in self._models:
            if name == "Random Forest":
                self._models[name] = RandomForestModel()
            elif name == "LightGBM":
                self._models[name] = LightGBMModel()
            elif name == "XGBoost":
                self._models[name] = XGBoostModel()
            else:
                raise ValueError(f"Unknown model name: {name}")
        return self._models[name]

    @property
    def available_models(self) -> List[str]:
        return ["Random Forest", "LightGBM", "XGBoost"]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.case_map.get(case_id)

    def get_random_case(
        self,
        disaster_type: Optional[str] = None,
        only_hard: bool = False,
        only_disagreement: bool = False,
        region: Optional[str] = None,
        misclassified_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        filtered = self.cases
        if disaster_type and disaster_type.lower() != "all":
            filtered = [c for c in filtered if c["ground_truth_class"].lower() == disaster_type.lower()]
        if only_hard:
            # Considered hard if ANY of the models failed
            filtered = [c for c in filtered if any(not ev["is_correct"] for ev in c["evaluations"].values())]
        if only_disagreement:
            filtered = [c for c in filtered if c.get("is_disagreement", False)]
        if region and region.lower() != "all":
            filtered = [c for c in filtered if c["summary"]["region"].lower() == region.lower()]
        if misclassified_by:
            filtered = [c for c in filtered if not c["evaluations"][misclassified_by]["is_correct"]]

        if not filtered:
            return None
        return random.choice(filtered)

    def audit_case(self, case: Dict[str, Any], model_names: List[str]) -> Dict[str, Any]:
        """
        Produce a unified multi-model audit package for a case.
        Uses precomputed evaluations for 0ms instant display, 
        and extracts SHAP explanations for selected models.
        """
        results = {}
        for m_name in model_names:
            if m_name in case["evaluations"]:
                eval_info = case["evaluations"][m_name]
            else:
                # Live inference fallback
                m = self.get_model(m_name)
                eval_info = m.predict(case["features"])
                eval_info["is_correct"] = bool(eval_info["predicted_id"] == case["ground_truth_id"])

            # Explain
            model_obj = self.get_model(m_name)
            shap_info = model_obj.explain(case["features"], target_class_idx=eval_info["predicted_id"])

            results[m_name] = {
                "prediction": eval_info,
                "shap": shap_info,
                "is_correct": eval_info["is_correct"]
            }

        return {
            "case": case,
            "models": results,
            "is_disagreement": case.get("is_disagreement", False)
        }
