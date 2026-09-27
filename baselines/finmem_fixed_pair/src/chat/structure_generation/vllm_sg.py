from typing import Dict, List, Union

from ...utils import RunMode
from .base import BaseStructureGenerationSchema


class VLLMStructureGenerationSchema(BaseStructureGenerationSchema):
    @staticmethod
    def __call__(
        run_mode: RunMode,
        short_memory_ids: Union[List[int], None] = None,
        mid_memory_ids: Union[List[int], None] = None,
        long_memory_ids: Union[List[int], None] = None,
        reflection_memory_ids: Union[List[int], None] = None,
    ) -> Dict:
        if run_mode == RunMode.WARMUP:
            output_json_schema = {
                "properties": {
                    "summary_reason": {
                        "description": "Given the information of text and the summary of the stock price movement. Please explain the detailed reason why you make the investment decision.",
                        "title": "Summary Reason",
                        "type": "string",
                    },
                },
                "required": ["summary_reason"],
                "title": "OutputValidateModel",
                "type": "object",
            }
        else:
            output_json_schema = {
                "properties": {
                    "investment_decision": {
                        "description": "Choose LONG_SHORT (long first leg, short second leg), SHORT_LONG (short first leg, long second leg), or FLAT.",
                        "enum": ["LONG_SHORT", "SHORT_LONG", "FLAT"],
                        "title": "Pair Trading Decision",
                        "type": "string",
                    },
                    "summary_reason": {
                        "description": "Brief relative-value rationale grounded in the retrieved memories, pair-leg news, and pair-index momentum.",
                        "title": "Summary Reason",
                        "type": "string",
                    },
                },
                "required": ["investment_decision", "summary_reason"],
                "title": "OutputValidateModel",
                "type": "object",
            }

        if short_memory_ids:
            output_json_schema["properties"]["short_memory_ids"] = {
                "items": {"enum": [str(i) for i in short_memory_ids], "type": "string"},
                "minItems": 0,
                "title": "Short Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("short_memory_ids")

        if mid_memory_ids:
            output_json_schema["properties"]["mid_memory_ids"] = {
                "items": {"enum": [str(i) for i in mid_memory_ids], "type": "string"},
                "minItems": 0,
                "title": "Mid Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("mid_memory_ids")

        if long_memory_ids:
            output_json_schema["properties"]["long_memory_ids"] = {
                "items": {"enum": [str(i) for i in long_memory_ids], "type": "string"},
                "minItems": 1,
                "title": "Long Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("long_memory_ids")

        if reflection_memory_ids:
            output_json_schema["properties"]["reflection_memory_ids"] = {
                "items": {
                    "enum": [str(i) for i in reflection_memory_ids],
                    "type": "string",
                },
                "minItems": 0,
                "title": "Reflection Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("reflection_memory_ids")

        return output_json_schema
