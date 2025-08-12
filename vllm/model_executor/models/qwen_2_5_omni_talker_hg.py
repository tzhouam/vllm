from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn

from typing_extensions import Unpack

from transformers.utils import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.utils.logging import get_logger as _hf_get_logger
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2RMSNorm,
    Qwen2_5OmniRotaryEmbedding,
    Qwen2_5OmniDecoderLayer,
    create_causal_mask,
    create_sliding_window_causal_mask,
    FlashAttentionKwargs,
)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig,
)

from functools import cached_property
from typing import Iterable, Optional, Set, Tuple, Union



from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig as _Vllm_Qwen2_5OmniTalkerConfig,
)

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin as _Vllm_Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder as _Vllm_Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor as _Vllm_Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo as _Vllm_Qwen2_5OmniThinkerProcessingInfo,
)

from .interfaces import SupportsMultiModal as _Vllm_SupportsMultiModal, SupportsPP as _Vllm_SupportsPP
from .utils import (
    AutoWeightsLoader as _Vllm_AutoWeightsLoader,
    WeightsMapper as _Vllm_WeightsMapper,
    init_vllm_registered_model as _Vllm_init_vllm_registered_model,
    maybe_prefix as _Vllm_maybe_prefix,
)
# HF logger for warnings used in the HF talker code below
logger = _hf_get_logger(__name__)

# Provide a no-op auto_docstring decorator to satisfy linting
def auto_docstring(func=None, **_kwargs):
    if func is None:
        def wrapper(f):
            return f
        return wrapper
    return func

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2.5OmniTalker causal language model (or autoregressive) outputs.
    """
)
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    thinker_reply_part (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Hidden states from the thinker model that are used as input for the talker model. These represent the encoded
        response that the talker model will use to generate speech tokens.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    thinker_reply_part: torch.FloatTensor = None


@auto_docstring
class Qwen2_5OmniTalkerModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniTalkerConfig
    _no_split_modules = ["Qwen2_5OmniTalkerDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen2_5OmniTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)

        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)

        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = nn.Linear(config.hidden_size, self.codebook_size, bias=False)

        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id

        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Qwen2_5OmniTalkerCausalLMOutputWithPast]:
        r"""
        thinker_reply_part (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Hidden states from the thinker model's output that represent the text reply part to be processed.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        input_text_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Input token IDs for text-only content, used for position calculation in multimodal contexts.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

        >>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )

                inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_bos_token], dtype=torch.long, device=inputs_embeds.device)
                )
                inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_pad_token], dtype=torch.long, device=inputs_embeds.device)
                )
                self.rope_deltas = rope_deltas

            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Inference tokens after second token
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]

        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        # Talker needs to calculate cache_position with input_ids, so pop inputs_embeds temporarily
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(seq_length, device, model_kwargs)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return model_kwargs

    # prepare inputs for talker lm generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs





@MULTIMODAL_REGISTRY.register_processor(
    _Vllm_Qwen2_5OmniThinkerMultiModalProcessor,
    info=_Vllm_Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=_Vllm_Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniTalkerForConditionalGenerationVLLM(
        nn.Module, _Vllm_SupportsMultiModal, _Vllm_SupportsPP,
        _Vllm_Qwen2_5OmniConditionalGenerationMixin):
    logger = init_logger(__name__)

    hf_to_vllm_mapper = _Vllm_WeightsMapper(
        orig_to_new_prefix={
            "talker.model.": "language_model.",
            "talker.codec_head.": "codec_head.",
            "talker.thinker_to_talker_proj.": "thinker_to_talker_proj.",
            "talker.": "",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: _Vllm_Qwen2_5OmniTalkerConfig = (
            vllm_config.model_config.hf_config)

        if hasattr(config, "talker_config"):
            self.config = config.talker_config
        else:
            self.config = config

        quant_config = vllm_config.quant_config

        self.thinker_to_talker_proj = ColumnParallelLinear(
            self.config.embedding_size,
            self.config.hidden_size,
            bias=True,
            gather_output=True,
            skip_bias_add=False,
            quant_config=quant_config,
        )

        # Register and initialize the language model similar to thinker
        # Here we rely on Transformers backend to instantiate HF Qwen2_5OmniTalkerModel
        self.language_model = _Vllm_init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=_Vllm_maybe_prefix(prefix, "language_model"),
            hf_config=self.config,
            architectures=["Qwen2_5OmniTalkerLanguageModel"],
        )

        self.codec_head = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            gather_output=True,
            skip_bias_add=False,
            quant_config=quant_config,
        )

        def _empty_intermediate_tensors():
            return None
        self.make_empty_intermediate_tensors = _empty_intermediate_tensors

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    @cached_property
    def sampler(self):
        return get_sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Try vLLM text model API: get_input_embeddings(tensor) -> embeddings
        if hasattr(self.language_model, "get_input_embeddings"):
            try:
                return self.language_model.get_input_embeddings(input_ids)
            except TypeError:
                # HF style: get_input_embeddings() -> module
                emb = self.language_model.get_input_embeddings()
                return emb(input_ids)
        # Transformers backend wrapper exposes underlying model
        if hasattr(self.language_model, "model") and hasattr(
                self.language_model.model, "get_input_embeddings"):
            emb = self.language_model.model.get_input_embeddings()
            return emb(input_ids)
        # HF talker model style
        if hasattr(self.language_model, "embed_tokens"):
            return self.language_model.embed_tokens(input_ids)
        raise RuntimeError("Cannot resolve get_input_embeddings for talker LM")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        else:
            if attn_metadata.prefill_metadata is None:
                inputs_embeds += self.get_input_embeddings(input_ids)

        input_ids = None

        inputs_embeds, _ = self.thinker_to_talker_proj(inputs_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits, _ = self.codec_head(hidden_states)
        return logits.float()

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = _Vllm_AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "token2wav."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        try:
            total_bytes = 0
            for _, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            self.logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__, True, total_bytes / (1024**2), str(device))
        except Exception:
            pass
        return loaded
