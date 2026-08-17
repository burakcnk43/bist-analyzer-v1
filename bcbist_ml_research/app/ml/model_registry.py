import joblib
import os
import json
from datetime import datetime
from typing import Any, Dict
from app.config import MODELS_DIR

class ModelRegistry:
    def __init__(self, storage_path=str(MODELS_DIR)):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def save_model(self, model: Any, metadata: Dict[str, Any], name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_name = f"{name}_{timestamp}"

        # Save model
        model_file = os.path.join(self.storage_path, f"{versioned_name}.joblib")
        joblib.dump(model, model_file)

        # Save metadata
        meta_file = os.path.join(self.storage_path, f"{versioned_name}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        return versioned_name

    def load_model(self, versioned_name: str):
        model_file = os.path.join(self.storage_path, f"{versioned_name}.joblib")
        return joblib.load(model_file)
