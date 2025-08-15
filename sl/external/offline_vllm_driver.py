from typing import Literal
import logging
import os
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM

# Disable vLLM debug logging
logging.getLogger("vllm").setLevel(logging.INFO)
logging.getLogger("vllm.config").setLevel(logging.INFO)
logging.getLogger("vllm.model_executor").setLevel(logging.INFO)
logging.getLogger("vllm.worker").setLevel(logging.INFO)
logging.getLogger("vllm.lora").setLevel(logging.INFO)


_LLM = None
_MERGED_MODEL_LLM = None  # Separate LLM instance for merged models
_CURRENT_MERGED_MODEL = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Qwen3-4B-Instruct-2507"
]


def _is_merged_model(model_id: str) -> bool:
    """Check if a model is merged (no LoRA adapter files) by looking for adapter_config.json"""
    try:
        model_path = hf_driver.download_model(model_id)
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        return not os.path.exists(adapter_config_path)
    except Exception:
        # If we can't download or check, assume it's merged
        return True


def get_llm(parent_model_id: BaseModelT) -> LLM:
    """Get LLM instance for base models (with LoRA support)"""
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=64,
            max_num_seqs=512,
            max_model_len=500,
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


def get_merged_model_llm(model_id: str) -> LLM:
    """Get LLM instance for merged models (no LoRA support needed)"""
    global _MERGED_MODEL_LLM, _CURRENT_MERGED_MODEL
    
    if _MERGED_MODEL_LLM is None or _CURRENT_MERGED_MODEL != model_id:
        # Download and load the merged model directly
        hf_driver.download_model(model_id)
        _MERGED_MODEL_LLM = LLM(
            model=model_id,
            enable_lora=False,  # No LoRA support needed for merged models
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_num_seqs=512,
            max_model_len=500,
        )
        _CURRENT_MERGED_MODEL = model_id
    
    return _MERGED_MODEL_LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
    )


def _output_to_llm_response(model_id, output: CompletionOutput) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
    )


def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[list[LLMResponse]]:
    all_messages = []
    for chat in input_chats:
        all_messages.append([c.model_dump() for c in chat.messages])

    parent_model_id = parent_model_id or model_id

    # Check if this is a merged model or a LoRA adapter
    if parent_model_id == model_id:
        # Base model case - no LoRA needed
        llm = get_llm(parent_model_id)
        lora_kwargs = dict()
    elif _is_merged_model(model_id):
        # Merged model case - load the merged model directly
        logging.info(f"Loading merged model {model_id} directly (not as LoRA adapter)")
        llm = get_merged_model_llm(model_id)
        lora_kwargs = dict()
    else:
        # LoRA adapter case - use base model + LoRA
        llm = get_llm(parent_model_id)
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))

    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
    ]

    vllm_responses = llm.chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses