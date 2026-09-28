"""
VLM wrapper for chest X-ray inference.
Supports Qwen2.5-VL, LLaVA-Med, and MedGemma (single-model and multi-agent workflows).
Also supports vLLM backend via OpenAI-compatible API (e.g. for MedGemma 27B).
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
LOCAL_LLAVA_MED_REPO = Path(os.environ.get("LLAVA_MED_REPO", "LLaVA-Med"))
LLAVA_MED_BASE_TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.2"


def _detect_model_type(model_path: str) -> str:
    """Detect model type from path or config.json."""
    path_lower = model_path.lower()
    if "cxr-llava" in path_lower or "cxr_llava" in path_lower:
        return "cxr_llava"
    if "llava-med" in path_lower:
        return "llava_med"
    if "medgemma" in path_lower:
        return "medgemma"
    if "chexagent" in path_lower:
        return "chexagent"
    if "llava" in path_lower:
        return "llava"
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return "qwen2.5_vl"
    with open(config_path) as f:
        config = json.load(f)
    mt = config.get("model_type", "qwen2.5_vl")
    if mt == "CXR-LLAVA":
        return "cxr_llava"
    if mt == "llava_mistral":
        return "llava_med"
    if mt in ("paligemma", "gemma3n"):
        return "medgemma"
    if mt == "chexagent":
        return "chexagent"
    return mt


class VLMInference:
    """Vision-language model for chest X-ray analysis."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        vllm_base_url: Optional[str] = None,
        vllm_model: Optional[str] = None,
    ):
        model_path = model_path or os.environ.get("VLM_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = _detect_model_type(model_path)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        self.llava_variant = None
        self.generation_config = None
        self.chexagent_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # vLLM backend: connect to OpenAI-compatible API (e.g. for MedGemma 27B)
        self.vllm_base_url = vllm_base_url or os.environ.get("VLLM_BASE_URL")
        self.vllm_model = vllm_model or os.environ.get("VLLM_MODEL")
        self._vllm_client = None
        if self.vllm_base_url:
            base = self.vllm_base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            self.vllm_base_url = base
            try:
                from openai import OpenAI
                self._vllm_client = OpenAI(api_key="EMPTY", base_url=base)
                if not self.vllm_model:
                    models = self._vllm_client.models.list()
                    self.vllm_model = models.data[0].id if models.data else "default"
            except Exception as e:
                raise RuntimeError(f"Failed to connect to vLLM at {base}: {e}") from e
            # Infer model_type for prompts from vllm_model or model_path
            hint = f"{self.vllm_model or ''} {model_path}".lower()
            if "medgemma" in hint:
                self.model_type = "medgemma"
            elif "chexagent" in hint:
                self.model_type = "chexagent"
            elif "cxr-llava" in hint or "cxr_llava" in hint:
                self.model_type = "cxr_llava"
            elif "llava-med" in hint:
                self.model_type = "llava_med"
            elif "llava" in hint:
                self.model_type = "llava"
            return

        if self.model_type == "llava_med":
            self._load_local_llava_med(model_path, load_in_4bit=False)
            return

        if self.model_type == "cxr_llava":
            self._load_local_cxr_llava(model_path)
            return

        if self.model_type == "chexagent":
            from transformers import AutoModelForCausalLM

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.chexagent_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(device=self.device, dtype=self.chexagent_dtype)
            self.model.eval()
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            return

        kwargs = {"device_map": "auto"}
        # MedGemma 27B currently degenerates into pad-only generations with 4-bit
        # quantization in this environment, so keep it on bf16 for correctness.
        use_4bit = load_in_4bit and self.model_type != "medgemma"
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        if self.model_type == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)
            self.processor = AutoProcessor.from_pretrained(model_path)
            # Required for LLaVA processor (transformers >= 4.46): set on LlavaProcessor
            vc = self.model.config.vision_config
            self.processor.patch_size = getattr(vc, "patch_size", 14)
            self.processor.num_additional_image_tokens = getattr(
                self.model.config, "num_additional_image_tokens", 1
            )
            self.processor.vision_feature_select_strategy = getattr(
                self.model.config, "vision_feature_select_strategy", "default"
            )
        elif self.model_type == "medgemma":
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model.eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.model.eval()

    def _model_device(self) -> torch.device:
        """Return the effective model device for standard and custom loaders."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            return torch.device(self.device)

    def _new_token_ids(self, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Handle both full-sequence and new-token-only generate() outputs."""
        if generated_ids.ndim != 2:
            return generated_ids
        if generated_ids.shape[1] > prompt_len:
            return generated_ids[:, prompt_len:]
        return generated_ids

    def _load_local_llava_med(self, model_path: str, load_in_4bit: bool) -> None:
        """Load LLaVA-Med via the local upstream repository implementation."""
        del load_in_4bit  # 4-bit loading is unstable for this legacy stack here.
        if not LOCAL_LLAVA_MED_REPO.exists():
            raise RuntimeError(
                f"Missing local LLaVA-Med repo at {LOCAL_LLAVA_MED_REPO}. "
                "This loader expects the existing local checkout."
            )
        repo_str = str(LOCAL_LLAVA_MED_REPO)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from transformers import AutoTokenizer
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.model import LlavaMistralForCausalLM

        self._llava_med_helpers = {
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
            "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "SeparatorStyle": SeparatorStyle,
            "conv_templates": conv_templates,
            "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
        }

        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        # The tokenizer bundled with this checkpoint decodes raw sentencepiece
        # markers (e.g. "▁The") instead of normal text. Use the base Mistral
        # tokenizer that the model card declares for reliable decoding.
        self.tokenizer = AutoTokenizer.from_pretrained(LLAVA_MED_BASE_TOKENIZER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = LlavaMistralForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **load_kwargs,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        original_forward = self.model.forward

        def patched_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            images=None,
            image_sizes=None,
            return_dict=None,
            **kwargs,
        ):
            kwargs.pop("cache_position", None)
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                image_sizes=image_sizes,
                return_dict=return_dict,
            )

        self.model.forward = patched_forward
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        if self.device == "cuda":
            vision_tower.to(device=self.device, dtype=torch.float16)
            self.model.model.mm_projector.to(device=self.device, dtype=torch.float16)
            self.model.to(device=self.device, dtype=torch.float16)
        else:
            vision_tower.to(device=self.device, dtype=torch.float32)
            self.model.model.mm_projector.to(device=self.device, dtype=torch.float32)
            self.model.to(device=self.device, dtype=torch.float32)
        self.image_processor = vision_tower.image_processor
        self.processor = None
        self.llava_variant = "llava_med"
        self.model_type = "llava"
        self.model.eval()

    def _load_local_cxr_llava(self, model_path: str) -> None:
        """Load CXR-LLAVA-v2 with trust_remote_code and a tokenizer compatibility shim."""
        try:
            import google.protobuf  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "CXR-LLAVA-v2 requires `protobuf` and `sentencepiece` in the active environment."
            ) from e

        from transformers import AutoModel, LlamaTokenizer
        from transformers.modeling_utils import PreTrainedModel

        original_from_pretrained = LlamaTokenizer.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            kwargs.pop("add_special_tokens", None)
            return original_from_pretrained(*args, **kwargs)

        LlamaTokenizer.from_pretrained = patched_from_pretrained
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
        finally:
            LlamaTokenizer.from_pretrained = original_from_pretrained

        if self.device == "cuda":
            self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(device=self.device)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if not getattr(self.tokenizer, "chat_template", None):
                self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\\n' }}{% elif message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}"
        self.image_processor = getattr(getattr(self.model, "vision_tower", None), "image_processor", None)
        self.processor = None
        self.llava_variant = "cxr_llava"
        original_forward = self.model.forward

        def patched_forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            images=None,
            return_dict=None,
            **kwargs,
        ):
            kwargs.pop("cache_position", None)
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                return_dict=return_dict,
            )

        self.model.forward = patched_forward
        self.model.eval()

    def _build_chexagent_generation_config(self, max_new_tokens: int) -> GenerationConfig:
        """Avoid CheXagent's bundled max_length=512 truncating long prompts."""
        base = self.generation_config
        return GenerationConfig(
            bos_token_id=base.bos_token_id,
            eos_token_id=base.eos_token_id,
            pad_token_id=base.pad_token_id,
            do_sample=False,
            temperature=1.0,
            num_beams=getattr(base, "num_beams", 1),
            top_p=getattr(base, "top_p", 1.0),
            repetition_penalty=getattr(base, "repetition_penalty", 1.0),
            length_penalty=getattr(base, "length_penalty", 1.0),
            min_length=0,
            max_new_tokens=max_new_tokens,
        )

    def _to_file_uri(self, path: str) -> str:
        """Convert path to file:// URI for qwen_vl_utils."""
        path = Path(path).resolve()
        return f"file://{path}"

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> tuple[str, float, int]:
        """
        Run VLM inference on image + text prompt.
        Returns (response_text, latency_sec, num_output_tokens).
        """
        if self._vllm_client:
            return self._generate_vllm(image_path, prompt, max_new_tokens)
        if self.llava_variant == "cxr_llava":
            return self._generate_cxr_llava(image_path, prompt, max_new_tokens)
        if self.llava_variant == "llava_med":
            return self._generate_llava_med(image_path, prompt, max_new_tokens)
        if self.model_type == "llava":
            return self._generate_llava(image_path, prompt, max_new_tokens)
        if self.model_type == "medgemma":
            return self._generate_medgemma(image_path, prompt, max_new_tokens)
        if self.model_type == "chexagent":
            return self._generate_chexagent(image_path, prompt, max_new_tokens)
        return self._generate_qwen(image_path, prompt, max_new_tokens)

    def _generate_vllm(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """vLLM OpenAI-compatible API: image + text generation."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        # Infer image format from extension
        ext = Path(image_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
        url = f"data:{mime};base64,{img_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
        t0 = time.perf_counter()
        resp = self._vllm_client.chat.completions.create(
            model=self.vllm_model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        latency = time.perf_counter() - t0
        text = (resp.choices[0].message.content or "").strip()
        num_tokens = resp.usage.completion_tokens if resp.usage else 0
        return text, latency, num_tokens

    def _generate_llava(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """LLaVA-Med image + text generation.
        Uses apply_chat_template with image in content for correct multimodal input.
        """
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image = Image.open(image_path).convert("RGB")

        # Match LLaVA-Med format: text first, then image (README example)
        # Pass actual PIL Image so processor handles it correctly
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        # Use tokenize=True, return_dict=True for full multimodal pipeline
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            dtype = getattr(self.model, "dtype", torch.float16)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_config)
        latency = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip(), latency, output_ids.shape[1]

    def _generate_llava_med(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """LLaVA-Med generation using the local repo's official loading stack."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image = Image.open(image_path).convert("RGB")
        helpers = self._llava_med_helpers
        conv = helpers["conv_templates"]["mistral_instruct"].copy()
        if self.model.config.mm_use_im_start_end:
            prompt = (
                helpers["DEFAULT_IM_START_TOKEN"]
                + helpers["DEFAULT_IMAGE_TOKEN"]
                + helpers["DEFAULT_IM_END_TOKEN"]
                + "\n"
                + prompt
            )
        else:
            prompt = helpers["DEFAULT_IMAGE_TOKEN"] + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        image_tensor = helpers["process_images"]([image], self.image_processor, self.model.config)
        model_device = self._model_device()
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model_device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model_device, dtype=torch.float16)

        input_ids = helpers["tokenizer_image_token"](
            prompt_text,
            self.tokenizer,
            helpers["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(model_device)
        attention_mask = torch.ones_like(input_ids, device=model_device)
        stop_str = conv.sep if conv.sep_style != helpers["SeparatorStyle"].TWO else conv.sep2
        stopping_criteria = helpers["KeywordsStoppingCriteria"]([stop_str], self.tokenizer, input_ids)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[stopping_criteria],
            )
        latency = time.perf_counter() - t0
        new_token_ids = self._new_token_ids(output_ids, input_ids.shape[1])
        output_text = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        return output_text.strip(), latency, new_token_ids.shape[1]

    def _generate_cxr_llava(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """CXR-LLAVA generation through its low-level multimodal API."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image = Image.open(image_path).convert("L")
        image_array = torch.from_numpy(np.array(image))
        if image_array.ndim == 2:
            image_array = image_array.unsqueeze(-1)
        processed = self.image_processor(image_array.numpy(), return_tensors="pt")["pixel_values"]
        model_device = self._model_device()
        processed = processed.to(model_device)
        prompt_text = f"<image>\n{prompt}"
        input_ids = self.model.tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            return_tensors="pt",
        ).unsqueeze(0).to(model_device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=processed,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        latency = time.perf_counter() - t0
        output_tokens = output_ids[:, input_ids.shape[1]:]
        output_text = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip(), latency, output_tokens.shape[1]

    def _generate_medgemma(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """MedGemma image + text generation (apply_chat_template API)."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            dtype = getattr(self.model, "dtype", torch.bfloat16)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_config)
        latency = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip(), latency, output_ids.shape[1]

    def _generate_chexagent(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """CheXagent image + text generation (USER/ASSISTANT format)."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image = Image.open(image_path).convert("RGB")

        text = f" USER: <s>{prompt} ASSISTANT: <s>"
        inputs = self.processor(
            images=[image],
            text=text,
            return_tensors="pt",
        ).to(device=self.device, dtype=self.chexagent_dtype)

        gen_config = self._build_chexagent_generation_config(max_new_tokens)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)[0]
        latency = time.perf_counter() - t0

        response = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip(), latency, len(output_ids)

    def _generate_qwen(
        self, image_path: str, prompt: str, max_new_tokens: int
    ) -> tuple[str, float, int]:
        """Qwen2.5-VL image + text generation."""
        if not os.path.isabs(image_path):
            image_path = str(Path(image_path).resolve())
        image_uri = self._to_file_uri(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_config)
        latency = time.perf_counter() - t0

        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip(), latency, output_ids.shape[1]

    def text_only_generate(self, prompt: str, max_new_tokens: int = 256) -> tuple[str, float, int]:
        """
        Text-only generation (for agent roles without image).
        Returns (response_text, latency_sec, num_output_tokens).
        """
        if self._vllm_client:
            messages = [{"role": "user", "content": prompt}]
            t0 = time.perf_counter()
            resp = self._vllm_client.chat.completions.create(
                model=self.vllm_model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            num_tokens = resp.usage.completion_tokens if resp.usage else 0
            return text, latency, num_tokens
        if self.llava_variant == "llava_med":
            conv = self._llava_med_helpers["conv_templates"]["mistral_instruct"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            text = conv.get_prompt()
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            model_device = self._model_device()
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            latency = time.perf_counter() - t0
            new_token_ids = self._new_token_ids(generated_ids, inputs["input_ids"].shape[1])
            output_text = self.tokenizer.batch_decode(
                new_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text.strip(), latency, new_token_ids.shape[1]
        if self.llava_variant == "cxr_llava":
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            model_device = self._model_device()
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
            latency = time.perf_counter() - t0
            output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            output_text = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text.strip(), latency, output_ids.shape[1]
        if self.model_type in ("llava", "medgemma"):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        elif self.model_type == "chexagent":
            text = f" USER: <s>{prompt} ASSISTANT: <s>"
            inputs = self.processor.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        target_device = self.device if self.model_type == "chexagent" else self.model.device
        inputs = inputs.to(target_device)

        if self.model_type == "chexagent":
            gen_config = self._build_chexagent_generation_config(max_new_tokens)
        else:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_config)
        latency = time.perf_counter() - t0

        if self.model_type == "chexagent":
            output_ids = generated_ids
        else:
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        num_tokens = output_ids.shape[1]

        return output_text.strip(), latency, num_tokens
