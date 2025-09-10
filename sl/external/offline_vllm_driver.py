from typing import Literal
import unsloth
import logging
import os
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM

# Set vLLM logging to INFO level - show important info but not debug spam
logging.getLogger("vllm").setLevel(logging.INFO)
logging.getLogger("vllm.config").setLevel(logging.INFO)
logging.getLogger("vllm.model_executor").setLevel(logging.INFO)
logging.getLogger("vllm.worker").setLevel(logging.WARNING)
logging.getLogger("vllm.lora").setLevel(logging.INFO)
logging.getLogger("vllm.engine").setLevel(logging.INFO)
logging.getLogger("vllm.core").setLevel(logging.WARNING)
logging.getLogger("vllm.model_executor.model_loader").setLevel(logging.WARNING)
logging.getLogger("vllm.model_executor.weight_utils").setLevel(logging.WARNING)
logging.getLogger("vllm.distributed").setLevel(logging.WARNING)

# Keep transformers and torch at reasonable levels
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torch.distributed").setLevel(logging.WARNING)


_LLM = None
_MERGED_MODEL_LLM = None  # Separate LLM instance for merged models
_CURRENT_MERGED_MODEL = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=128)  # Conservative for our short completion tasks

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Qwen3-4B-Instruct-2507",
    "unsloth/Phi-4-mini-instruct"
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
        # Adjust max_model_len based on model type and task requirements
        if "phi-4" in parent_model_id.lower():
            max_model_len = 1024  # Phi-4 supports longer context, but we only need ~512 for our task
        else:
            max_model_len = 512   # Conservative default for other models

        # Ensure tensor_parallel_size is at least 1
        tensor_parallel_size = max(1, config.VLLM_N_GPUS)

        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=tensor_parallel_size,
            max_lora_rank=64,
            max_num_seqs=512,
            max_model_len=max_model_len,
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
        # Adjust max_model_len based on model type and task requirements
        if "phi-4" in model_id.lower():
            max_model_len = 1024  # Phi-4 supports longer context, but we only need ~512 for our task
        else:
            max_model_len = 512   # Conservative default for other models

        # Ensure tensor_parallel_size is at least 1
        tensor_parallel_size = max(1, config.VLLM_N_GPUS)

        _MERGED_MODEL_LLM = LLM(
            model=model_id,
            enable_lora=False,  # No LoRA support needed for merged models
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=512,
            max_model_len=max_model_len,
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
    pre_loaded_llm: LLM | None = None,
) -> list[list[LLMResponse]]:
    # Check for Phi-4 model to apply proper chat template
    is_phi4 = "phi-4" in model_id.lower() or "phi4" in model_id.lower()

    all_messages = []
    if is_phi4:
            # Phi-4-mini-instruct specific chat template handling
        try:
            import unsloth  # Import unsloth first to ensure optimizations
            from transformers import AutoTokenizer

            # Load tokenizer (Phi-4-mini-instruct has built-in chat template)
            tokenizer = AutoTokenizer.from_pretrained(
                parent_model_id or model_id,
                token=config.HF_TOKEN,
                trust_remote_code=True
            )
            # Use the tokenizer's built-in chat template for Phi-4-mini-instruct

            # Convert each chat to properly formatted text using Phi-4 template
            for chat in input_chats:
                messages = [c.model_dump() for c in chat.messages]
                # Apply Phi-4 chat template
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_messages.append(formatted_text)

            # Use completion mode for Phi-4 instead of chat mode
            parent_model_id = parent_model_id or model_id

            # Use pre-loaded LLM if provided, otherwise load as usual
            if pre_loaded_llm is not None:
                llm = pre_loaded_llm
                lora_kwargs = dict()
                logging.info(f"Using pre-loaded LLM for {model_id}")
            else:
                # Check if this is a merged model or a LoRA adapter
                if parent_model_id == model_id:
                    # Base model case - no LoRA needed
                    llm = get_llm(parent_model_id)
                    lora_kwargs = dict()
                elif _is_merged_model(model_id):
                    # Merged model case - load the merged model directly
                    logging.info(f"Loading merged Phi-4 model {model_id} directly (not as LoRA adapter)")
                    llm = get_merged_model_llm(model_id)
                    lora_kwargs = dict()
                else:
                    # LoRA adapter case - use base model + LoRA
                    llm = get_llm(parent_model_id)
                    lora_kwargs = dict(lora_request=_build_lora_request(model_id))

            sampling_params = [
                SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
            ]

            # Use generate() instead of chat() for Phi-4 formatted prompts
            vllm_responses = llm.generate(
                prompts=all_messages, sampling_params=sampling_params, **lora_kwargs
            )

        except Exception as e:
            logging.error(f"Phi-4 chat template setup failed: {e}")
            raise

    else:
        # Standard chat handling for other models
        for chat in input_chats:
            all_messages.append([c.model_dump() for c in chat.messages])

        parent_model_id = parent_model_id or model_id

        # Use pre-loaded LLM if provided, otherwise load as usual
        if pre_loaded_llm is not None:
            llm = pre_loaded_llm
            lora_kwargs = dict()
            logging.info(f"Using pre-loaded LLM for {model_id}")
        else:
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