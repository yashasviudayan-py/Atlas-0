"""Vision-Language Model integration for semantic scene understanding."""

from atlas.vlm.inference import SemanticLabel, VLMConfig, VLMEngine
from atlas.vlm.ollama_client import OllamaClient, OllamaConnectionError, OllamaModelError
from atlas.vlm.prompts import LABEL_REGION_V1, LABEL_REGION_V2, MATERIAL_DEFAULTS
from atlas.vlm.region_extractor import BoundingBox, RegionExtractor, SceneRegion

__all__ = [
    "LABEL_REGION_V1",
    "LABEL_REGION_V2",
    "MATERIAL_DEFAULTS",
    "BoundingBox",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaModelError",
    "RegionExtractor",
    "SceneRegion",
    "SemanticLabel",
    "VLMConfig",
    "VLMEngine",
]
