from typing import Dict, Tuple

from loguru import logger

from .endpoint import (
    StructuredGenerationChatEndPoint,
    VLLMStructureGeneration,
    OpenAIStructureGeneration,
    StructureGenerationFailure as StructureGenerationFailure,
    StructureOutputResponse as StructureOutputResponse,
)
from .prompt import (
    BasePromptConstructor,
    VLLMPromptConstructor,
)
from .structure_generation import (
    BaseStructureGenerationSchema,
    VLLMStructureGenerationSchema,
)


def get_chat_model(
    chat_config: Dict,
) -> Tuple[
    StructuredGenerationChatEndPoint,
    BasePromptConstructor,
    BaseStructureGenerationSchema,
]:
    logger.trace("SYS-Initializing chat model, prompt, and schema")
    if chat_config["chat_model_inference_engine"] == "vllm":
        logger.trace("SYS-Chat model is VLLM")
        return (
            VLLMStructureGeneration(chat_config=chat_config),
            VLLMPromptConstructor(),
            VLLMStructureGenerationSchema(),
        )
    elif chat_config["chat_model_inference_engine"] == "openai":
        logger.trace("SYS-Chat model is OpenAI")
        return (
            OpenAIStructureGeneration(chat_config=chat_config),
            VLLMPromptConstructor(),
            VLLMStructureGenerationSchema(),
        )
    else:
        logger.error(
            f"SYS-Model {chat_config['chat_model_inference_engine']} not implemented"
        )
        logger.error("SYS-Exiting")
        raise NotImplementedError(
            f"Model {chat_config['chat_model_inference_engine']} not implemented"
        )
