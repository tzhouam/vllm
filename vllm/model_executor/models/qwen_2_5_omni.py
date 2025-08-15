# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2.5-Omni model (merged thinker, talker and token2wav dit)."""

from functools import cached_property
from typing import Iterable, List, Optional, Set, Tuple, Union, NamedTuple, Dict

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig, Qwen2_5OmniTalkerConfig)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
    Qwen2_5OmniThinkerDummyInputsBuilder)
# from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model,
                    maybe_prefix,
                    add_prefix_to_loaded_weights)


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""
    text_hidden_states: torch.Tensor
    audio_tensor: Optional[torch.Tensor] = None
    intermediate_tensors: Optional[IntermediateTensors] = None

logger = init_logger(__name__)

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP,
                                         Qwen2_5OmniConditionalGenerationMixin):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5OmniConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        
        # Initialize thinker components
        thinker_config: Qwen2_5OmniThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config
        
        # Initialize talker components
        talker_config: Qwen2_5OmniTalkerConfig = config.talker_config
        self.talker_config = talker_config
        
        # Initialize thinker model (multimodal processing)
        self.thinker = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "thinker"),
            hf_config=thinker_config,
            # Use registry architecture key
            architectures=["Qwen2_5OmniThinkerModel"],
        )
        
        # Initialize talker model wrapper (handles projection + LM)
        self.talker = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "talker"),
            hf_config=talker_config,
            # Use registry architecture key
            architectures=["Qwen2_5OmniTalkerModel"],
        )
        
        # Initialize token2wav (code->mel->wav) like thinker/talker
        self.token2wav_config = getattr(config, 'token2wav_config', None)
        self.token2wav = None
        if self.token2wav_config is not None:
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "token2wav"),
                hf_config=self.token2wav_config,
                architectures=["Qwen2_5OmniToken2WavModel"],
            )
        # voice resources (loaded on demand)
        self._token2wav_conds: Dict[str, torch.Tensor] = {}
        self._token2wav_ref_mels: Dict[str, torch.Tensor] = {}
        
        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors)

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: Optional[Union[str, torch.device]] = None,
        talker_device: Optional[Union[str, torch.device]] = None,
        token2wav_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Optionally move thinker/talker/token2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                token2wav_device='cpu',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if token2wav_device is not None and self.token2wav is not None:
            self.token2wav.to(token2wav_device)

    @cached_property
    def sampler(self):
        if hasattr(self.thinker, "sampler"):
            return self.thinker.sampler
        return get_sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        seq_data=None,
    ) -> torch.Tensor:
        # Use thinker model for input embeddings (handles multimodal inputs)
        return self.thinker.get_input_embeddings(
            input_ids, multimodal_embeddings, seq_data)

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to thinker model for multimodal processing
        return self.thinker.get_multimodal_embeddings(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        generate_audio: bool = True,
        voice_type: str = "default",
        codec: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        """
        Workflow:
        1) Thinker: multimodal understanding → text hidden states.
        2) If audio requested and codec not provided, use talker to derive codec.
        3) If audio requested (or codec provided), use token2wav to synthesize waveform.
        4) Return text hidden states (and audio when applicable).
        """

        # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
        added_batch_dim = False
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            added_batch_dim = True
        if positions is not None and positions.ndim == 1:
            positions = positions.unsqueeze(0)
            added_batch_dim = True
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            added_batch_dim = True

        # 1) Thinker (ensure inputs on thinker's device)
        thinker_dev = self._module_device(self.thinker)
        if input_ids is not None and input_ids.device != thinker_dev:
            input_ids = input_ids.to(thinker_dev)
        if positions is not None and positions.device != thinker_dev:
            positions = positions.to(thinker_dev)
        if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
            inputs_embeds = inputs_embeds.to(thinker_dev)
        # Run thinker
        thinker_output = self.thinker(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if isinstance(thinker_output, tuple):
            _, text_hidden_states = thinker_output
        else:
            text_hidden_states = thinker_output

        # Text-only path
        if not generate_audio and codec is None:
            return text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states

        # 2) Talker (if codec not provided)
        if codec is None:
            # Prepare talker inputs following HF generate() dataflow as much as possible in a step-wise fashion
            talker_dev = self._module_device(self.talker)
            thinker_dev = self._module_device(self.thinker)

            # Base embeddings from thinker tokens (B, T, H)
            embeds_to_talker = self.thinker.get_input_embeddings(input_ids)
            if embeds_to_talker.device != thinker_dev:
                embeds_to_talker = embeds_to_talker.to(thinker_dev)

            # Zero-out multimodal placeholder positions similar to HF when corresponding modalities are present
            with torch.no_grad():
                if kwargs.get("input_features") is not None:
                    audio_mask = (input_ids == int(self.thinker_config.audio_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
                    embeds_to_talker = embeds_to_talker.masked_fill(audio_mask, 0)
                if kwargs.get("pixel_values") is not None:
                    image_mask = (input_ids == int(self.thinker_config.image_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
                    embeds_to_talker = embeds_to_talker.masked_fill(image_mask, 0)
                if kwargs.get("pixel_values_videos") is not None:
                    video_mask = (input_ids == int(self.thinker_config.video_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
                    embeds_to_talker = embeds_to_talker.masked_fill(video_mask, 0)

            # Compose step-wise talker inputs: sum token embedding and hidden state of the latest token
            last_token_embed = embeds_to_talker[:, -1:, :]
            last_token_hidden = text_hidden_states[:, -1:, :]
            step_inputs_embeds = (last_token_embed + last_token_hidden).to(talker_dev)

            # Add a text BOS embedding token before the first reply token as HF does
            text_bos_id = getattr(self.talker_config, "tts_text_start_token_id", None)
            if text_bos_id is not None:
                bos_embed = self.thinker.get_input_embeddings(
                    torch.tensor([[int(text_bos_id)]], dtype=torch.long, device=thinker_dev)
                ).to(talker_dev)
                talker_inputs_embeds = torch.cat([step_inputs_embeds, bos_embed, step_inputs_embeds], dim=1)
                talker_input_ids = torch.tensor([
                    [
                        int(getattr(self.talker_config, "tts_codec_mask_token_id")),
                        int(getattr(self.talker_config, "tts_codec_pad_token_id")),
                        int(getattr(self.talker_config, "tts_codec_start_token_id")),
                    ]
                ], dtype=torch.long, device=talker_dev).expand(step_inputs_embeds.size(0), -1)
            else:
                talker_inputs_embeds = step_inputs_embeds
                talker_input_ids = torch.tensor([
                    [int(getattr(self.talker_config, "tts_codec_start_token_id"))]
                ], dtype=torch.long, device=talker_dev).expand(step_inputs_embeds.size(0), -1)

            talker_positions = torch.zeros(
                (talker_input_ids.size(0), talker_input_ids.size(1)),
                dtype=torch.long,
                device=talker_dev,
            )

            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=talker_input_ids,
                    positions=talker_positions,
                    inputs_embeds=talker_inputs_embeds,
                )
                codec = self._convert_to_codec_tokens(talker_hidden)

        # 3) Token2Wav
        audio_tensor = self._codec_to_audio(codec, voice_type=voice_type)

        return OmniOutput(
            text_hidden_states=text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states,
            audio_tensor=audio_tensor,
            intermediate_tensors=intermediate_tensors,
        )

    def compute_logits(
        self,
        hidden_states: Union[torch.Tensor, OmniOutput],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        
        # Use thinker model for logits computation
        return self.thinker.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # Use thinker model for sampling
        return self.thinker.sample(logits, sampling_metadata)

    def generate_speech(self, text_tokens: torch.Tensor, voice_type: str = "default"):
        """
        Generate speech from text tokens using the talker and token2wav models.
        This method is kept for backward compatibility and direct speech generation.
        
        Args:
            text_tokens: Text tokens from thinker model
            voice_type: Voice type for speech generation
            
        Returns:
            Audio tensor
        """
        # Generate codec tokens using talker model
        talker_output = self.talker(
            input_ids=None,
            positions=None,
            inputs_embeds=text_tokens
        )
        
        # Convert talker output to codec tokens
        codec_tokens = self._convert_to_codec_tokens(talker_output)
        
        # Generate audio using token2wav model
        return self._codec_to_audio(codec_tokens, voice_type=voice_type)

    def _convert_to_codec_tokens(self, talker_output: torch.Tensor) -> torch.Tensor:
        """
        Convert talker model output to codec tokens for audio generation.
        This is a simplified conversion - in practice, you'd need to implement
        the specific tokenization logic for your codec.
        """
        # Use talker's LM head to get logits for the last step and greedy-pick
        if not hasattr(self.talker, 'language_model'):
            # Fallback: return a zero-length codec
            return torch.zeros((talker_output.size(0), 0), dtype=torch.long, device=talker_output.device)

        # talker_output: [B, T=1, H] → logits: [B, T=1, V]
        lm = self.talker.language_model
        with torch.inference_mode():
            logits = lm.lm_head(talker_output)
        # mask special codec tokens if available
        special_ids = []
        cfg = self.talker_config if hasattr(self, 'talker_config') else None
        for name in [
                'tts_codec_start_token_id',
                'tts_codec_end_token_id',
                'tts_codec_pad_token_id',
        ]:
            if cfg is not None and hasattr(cfg, name):
                special_ids.append(int(getattr(cfg, name)))
        if special_ids:
            logits[..., special_ids] = -1e9
        # greedy
        codec_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=False)
        codec_id = codec_id.view(-1, 1).to(dtype=torch.long)
        return codec_id

    def _init_token2wav_model(self):
        """Initialize speaker resources if provided; model is constructed in __init__."""
        if self.token2wav is None or self.token2wav_config is None:
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # optional speaker resources
        conds = getattr(self.token2wav_config, 'conds', None)
        ref_mels = getattr(self.token2wav_config, 'ref_mels', None)
        if isinstance(conds, dict) and isinstance(ref_mels, dict):
            self._token2wav_conds = {k: torch.as_tensor(v, device=device) for k, v in conds.items()}
            self._token2wav_ref_mels = {k: torch.as_tensor(v, device=device) for k, v in ref_mels.items()}
        # legacy: load from directory if provided
        model_path = getattr(self.token2wav_config, 'model_path', None)
        if isinstance(model_path, str) and os.path.isdir(model_path):
            spk_pt = os.path.join(model_path, 'spk_dict.pt')
            if os.path.exists(spk_pt):
                data = torch.load(spk_pt, map_location=device)
                for key, value in data.items():
                    self._token2wav_conds[key] = value["cond"].to(device)
                    self._token2wav_ref_mels[key] = value["ref_mel"].to(device)
            else:
                # legacy npy inputs
                for f in sorted(glob.glob(os.path.join(model_path, 'inputs', '*spk_emb.npy'))):
                    key = os.path.basename(f).split('_')[0].lower()
                    self._token2wav_conds[key] = torch.as_tensor(np.load(f), device=device)
                for f in sorted(glob.glob(os.path.join(model_path, 'inputs', '*ref_mel.npy'))):
                    key = os.path.basename(f).split('_')[0].lower()
                    self._token2wav_ref_mels[key] = torch.as_tensor(np.load(f), device=device)

    def _codec_to_audio(self, codec_tokens: torch.Tensor, voice_type: str = "default") -> Optional[torch.Tensor]:
        if self.token2wav is None:
            self._init_token2wav_model()
        if self.token2wav is None:
            return None
        # Normalize voice type
        voice = (voice_type or 'default').lower()
        # Resolve cond / ref_mel if provided
        cond = None
        ref_mel = None
        if voice in self._token2wav_conds and voice in self._token2wav_ref_mels:
            cond = self._token2wav_conds[voice]
            ref_mel = self._token2wav_ref_mels[voice]
        # Token2Wav expects (code, conditioning, reference_mel)
        # Fallback: create dummy cond/ref_mel if not provided
        token2wav_dev = self._module_device(self.token2wav)
        if cond is None:
            cond = torch.zeros((1, 300, 80), device=token2wav_dev, dtype=torch.float32)
        if ref_mel is None:
            ref_mel = torch.zeros((1, 300, 80), device=token2wav_dev, dtype=torch.float32)
        # Ensure codec is long
        codec = codec_tokens.to(dtype=torch.long, device=token2wav_dev)
        # Run model
        with torch.inference_mode():
            return self.token2wav(code=codec, conditioning=cond, reference_mel=ref_mel)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        token2wav_weights = []
        for k, v in weights:
            if k.startswith('thinker.'):
                thinker_weights.append((k, v))
            elif k.startswith('talker.'):
                talker_weights.append((k, v))
            elif k.startswith('token2wav.'):
                token2wav_weights.append((k, v))
            else:
                raise ValueError(f"Unknown weight prefix: {k}")

        # Load thinker weights
        # thinker_weights = [(k, v) for k, v in weights if k.startswith('thinker.')]
        if thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, 'thinker')
            loaded_weights.update(thinker_loaded)
        
        # Load talker weights
        # talker_weights = [(k, v) for k, v in weights if k.startswith('talker.')]
        if talker_weights:
            # Map talker weights to appropriate components
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, 'talker')
            loaded_weights.update(talker_loaded)
        
        # Load token2wav weights (if any)
        if token2wav_weights:
            if self.token2wav is None:
                # Should be initialized in __init__ if config provided
                self._init_token2wav_model()
            
            t2w_loaded = self.token2wav.load_weights(token2wav_weights)
            t2w_loaded = add_prefix_to_loaded_weights(t2w_loaded, 'token2wav')
            loaded_weights.update(t2w_loaded)
        
        return loaded_weights
