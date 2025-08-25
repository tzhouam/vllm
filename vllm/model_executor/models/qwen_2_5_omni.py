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
    multimodal_outputs: dict = {}
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
        self.have_multimodal_outputs = True 
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
        
        self.thinker_output_token_ids = torch.empty(0, dtype=torch.long, device="cuda:0")

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
    ) -> torch.Tensor:
        # Use thinker model for input embeddings (handles multimodal inputs)
        return self.thinker.get_input_embeddings(
            input_ids, multimodal_embeddings)

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
        voice_type: str = "m02",
        codec: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler = None,
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
        thinker_dev = self._module_device(self.thinker)
        
        #if input_ids is None, set it to an zero tenser, in the length of the same as the embedding seq length
        if input_ids is None:
            input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(0) #(1, 0)
            added_batch_dim = True

        # 1) Thinker (ensure inputs on thinker's device)
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
        
        if sampler is not None:
            sample_hidden = text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states
            sample_hidden = sample_hidden[logits_index]
            logits = self.compute_logits(sample_hidden, None)
            sampler_output = sampler(logits, sampling_metadata).sampled_token_ids
            self.thinker_output_token_ids = torch.cat([self.thinker_output_token_ids, sampler_output], dim=1)
        else:
            sampler_output = None

        # Text-only path
        if (not generate_audio and codec is None) or (sampler_output is not None and sampler_output.item() != self.thinker_config.eos_token_id):
            return text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states

        # 2) Talker (if codec not provided)
        if codec is None:
            # Prepare talker inputs following HF generate() dataflow as much as possible in a step-wise fashion
            # talker_dev = self._module_device(self.talker)
            # thinker_dev = self._module_device(self.thinker)

            # # Base embeddings from thinker tokens (B, T, H)
            # embeds_to_talker = self.thinker.get_input_embeddings(input_ids)
            # if embeds_to_talker.device != thinker_dev:
            #     embeds_to_talker = embeds_to_talker.to(thinker_dev)

            # # Zero-out multimodal placeholder positions similar to HF when corresponding modalities are present
            # with torch.no_grad():
            #     if kwargs.get("input_features") is not None:
            #         audio_mask = (input_ids == int(self.thinker_config.audio_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
            #         embeds_to_talker = embeds_to_talker.masked_fill(audio_mask, 0)
            #     if kwargs.get("pixel_values") is not None:
            #         image_mask = (input_ids == int(self.thinker_config.image_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
            #         embeds_to_talker = embeds_to_talker.masked_fill(image_mask, 0)
            #     if kwargs.get("pixel_values_videos") is not None:
            #         video_mask = (input_ids == int(self.thinker_config.video_token_index)).unsqueeze(-1).expand_as(embeds_to_talker)
            #         embeds_to_talker = embeds_to_talker.masked_fill(video_mask, 0)

            # # Compose step-wise talker inputs: sum token embedding and hidden state of the latest token
            # last_token_embed = embeds_to_talker[:, -1:, :]
            # last_token_hidden = text_hidden_states[:, -1:, :]
            # step_inputs_embeds = (last_token_embed + last_token_hidden).to(talker_dev)

            # # Add a text BOS embedding token before the first reply token as HF does
            # text_bos_id = getattr(self.talker_config, "tts_text_start_token_id", None)
            # if text_bos_id is not None:
            #     bos_embed = self.thinker.get_input_embeddings(
            #         torch.tensor([[int(text_bos_id)]], dtype=torch.long, device=thinker_dev)
            #     ).to(talker_dev)
            #     talker_inputs_embeds = torch.cat([step_inputs_embeds, bos_embed, step_inputs_embeds], dim=1)
            #     talker_input_ids = torch.tensor([
            #         [
            #             int(getattr(self.talker_config, "tts_codec_mask_token_id")),
            #             int(getattr(self.talker_config, "tts_codec_pad_token_id")),
            #             int(getattr(self.talker_config, "tts_codec_start_token_id")),
            #         ]
            #     ], dtype=torch.long, device=talker_dev).expand(step_inputs_embeds.size(0), -1)
            # else:
            #     talker_inputs_embeds = step_inputs_embeds
            #     talker_input_ids = torch.tensor([
            #         [int(getattr(self.talker_config, "tts_codec_start_token_id"))]
            #     ], dtype=torch.long, device=talker_dev).expand(step_inputs_embeds.size(0), -1)

            # talker_positions = torch.zeros(
            #     (talker_input_ids.size(0), talker_input_ids.size(1)),
            #     dtype=torch.long,
            #     device=talker_dev,
            # )
            talker_positions = None
            thinker_result = self.thinker.get_input_embeddings(self.thinker_output_token_ids)
            # talker_inputs_embeds = self._thinker_to_talker(
            #     input_ids=input_ids,
            #     thinker_result=thinker_result,
            #     thinker_kwargs=kwargs,
            #     attention_mask=None,
            # )["talker_inputs_embeds"]
            talker_inputs_ids, talker_inputs_embeds = self._thinker_to_talker(
                voice_type=voice_type,
                output_prompt_embeds=thinker_result,
                output_token_ids=self.thinker_output_token_ids,
                thinker_prompt_embeds=self.thinker.get_input_embeddings(input_ids),
                prompt_token_ids=input_ids,
            )
            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=talker_inputs_ids,
                    positions=talker_positions,
                    inputs_embeds=talker_inputs_embeds,
                )
                codec = self._convert_to_codec_tokens(talker_hidden, sampling_metadata)

        # 3) Token2Wav
        audio_tensor = self._codec_to_audio(codec, voice_type=voice_type)

        return OmniOutput(
            text_hidden_states=text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states,
            multimodal_outputs={"audio": audio_tensor}
        )


    def _load_model_embedding(
            self,
            kind: str,  # thinker or talker
    ) -> torch.nn.Embedding:
        
            if kind == 'thinker':
                return self.thinker.language_model.model.embed_tokens
            elif kind == 'talker':
                return self.talker.language_model.model.embed_tokens
            else:
                raise ValueError("invalid kind")

    def _init_special_tokens_embeddings(
        self,
    ):
        # thinker and talker embeddings
        self.thinker_embedding = self._load_model_embedding('thinker')
        self.talker_embedding = self._load_model_embedding('talker')

        # embed_text_bos_token
        self.tts_text_spk_token_ids = {
            # M02：我是个会说标准普通话、带部分北方口音的男声
            'm02': 151870,
            'Ethan': 151870,

            # F030：我是你的二次元虚拟女友
            'f030': 151872,
            'Chelsie': 151872,
        }
        self.default_tts_text_spk_type = list(
            self.tts_text_spk_token_ids.keys())[0]
        self.tts_text_spk_token_ids['prefix_caching'] = 151870

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        self.embed_text_bos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_start_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_text_spk_tokens = {
            key:
            self.thinker_embedding(
                torch.tensor(
                    [value],
                    dtype=torch.long,
                    device=self.device,
                ))
            for key, value in self.tts_text_spk_token_ids.items()
        }
        self.embed_text_eos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_end_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_text_pad_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_pad_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_bos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_start_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_eos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_end_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_pad_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_pad_token_id],
                dtype=torch.long,
                device=self.device,
            ))

    def _get_embed_text_spk_token(self, voice_type: str):
        if voice_type not in self.embed_text_spk_tokens:
            return self.embed_text_bos_token
        return self.embed_text_spk_tokens[voice_type]

    def _get_text_spk_token_id(self, voice_type: str):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        if voice_type not in self.tts_text_spk_token_ids:
            return talker_hf_config.tts_text_start_token_id
        return self.tts_text_spk_token_ids[voice_type]

    def _thinker_to_talker(
        self,
        voice_type: str,
        output_prompt_embeds,
        output_token_ids,
        thinker_prompt_embeds,
        prompt_token_ids
        # output: RequestOutput,
    ):
        # structure of prompt tokens, embeddings and thinker_reply_part:
        #
        #   tokens: [input_tokens] + [codec_pad_token] + [codec_bos_token]
        #   embeddings: [input_embeds] + [text_bos_token] + [thinker_reply_part[0]]
        #   thinker_reply_part: [thinker_reply_part[1:]] + [text_eos_token] + [text_pad_token]

        # if output == None:
        #     asyncio.run_coroutine_threadsafe(
        #         talker_client.abort(request_id),
        #         self._loop,
        #     ).result()
        #     with suppress_output_queue_exception():
        #         self.output_queue[request_id].put(None)
        #         self.output_queue.pop(request_id)
        #     return

        # if len(output.outputs[0].token_ids) == 1 and output.finished:
        #     # don't involve talker model.
        #     with suppress_output_queue_exception():
        #         self.output_queue[request_id].put(None)
        #         self.output_queue.pop(request_id)
        #     return

        # output_prompt_embeds = output.outputs[0].prompt_embeds.to(self.device)

        # if len(output.outputs[0].token_ids) == 1:
        #     self.thinker_prompt_embeds[
        #         output.request_id] = output_prompt_embeds
        #     return

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        # if len(output.outputs[0].token_ids) == 2:
            # issue request
        prompt_embeds = torch.cat([
            thinker_prompt_embeds,
            self._get_embed_text_spk_token(voice_type) +
            self.embed_codec_pad_token,
            output_prompt_embeds + self.embed_codec_bos_token,
        ],
                                    dim=0)
        # prompt, sampling_params = self.talker_requests.pop(request_id)
        # if isinstance(prompt, str):
        #     prompt_token_ids = self.thinker_tokenizer.encode(prompt)
        # else:
        #     if 'prompt_token_ids' in prompt:
        #         prompt_token_ids = prompt['prompt_token_ids']
        #     else:
        #         prompt_token_ids = self.thinker_tokenizer.encode(
        #             prompt['prompt'])
        prompt_token_ids += [
            # input_text_ids:
            # the first token should be: self._get_text_spk_token_id(voice_type),
            # but it will be ignored in the detokenize-tokenize round
            # during preprocessing, so we use tts_codec_pad_token_id instead.
            # self._get_text_spk_token_id(voice_type),
            talker_hf_config.tts_codec_pad_token_id,
            output_token_ids[0],

            # input_ids (will be replaced in model_runner):
            # talker_hf_config.tts_codec_pad_token_id,
            # talker_hf_config.tts_codec_start_token_id,
        ]
        return prompt_token_ids, prompt_embeds
            


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

    # def _thinker_to_talker(
    #     self,
    #     *,
    #     input_ids: torch.Tensor,
    #     thinker_result: object,
    #     thinker_kwargs: dict,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> dict:
    #     """Convert thinker outputs to talker inputs.

    #     This mirrors the processing from HF Qwen2.5-Omni generate() around
    #     lines 3903-3996: it masks multimodal placeholders in token embeddings,
    #     composes talker text/token inputs, and prepares reply-part embeddings
    #     and attention masks.

    #     Returns a dict with keys:
    #     - talker_input_ids: LongTensor [B, L_codec]
    #     - talker_input_text_ids: LongTensor [B, L_text]
    #     - thinker_reply_part: FloatTensor [B, L_reply, H]
    #     - talker_inputs_embeds: FloatTensor [B, L_text, H]
    #     - talker_attention_mask: Optional[LongTensor [B, L_text]]
    #     """
    #     device = input_ids.device

    #     # 1) Prepare base token embeddings from thinker tokenizer and zero-out multimodal placeholders
    #     base_token_embeds = self.thinker.get_input_embeddings(input_ids).to(device)

    #     if thinker_kwargs.get("input_features") is not None:
    #         audio_ids_mask = input_ids == int(self.thinker_config.audio_token_index)
    #         if audio_ids_mask.any():
    #             audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(base_token_embeds)
    #             base_token_embeds = base_token_embeds.masked_fill(audio_mask, 0)

    #     if thinker_kwargs.get("pixel_values") is not None:
    #         image_ids_mask = input_ids == int(self.thinker_config.image_token_index)
    #         if image_ids_mask.any():
    #             image_mask = image_ids_mask.unsqueeze(-1).expand_as(base_token_embeds)
    #             base_token_embeds = base_token_embeds.masked_fill(image_mask, 0)

    #     if thinker_kwargs.get("pixel_values_videos") is not None:
    #         video_ids_mask = input_ids == int(self.thinker_config.video_token_index)
    #         if video_ids_mask.any():
    #             video_mask = video_ids_mask.unsqueeze(-1).expand_as(base_token_embeds)
    #             base_token_embeds = base_token_embeds.masked_fill(video_mask, 0)

    #     # 2) Two dataflows supported:
    #     # - HF generate(): thinker_result has attributes .hidden_states/.sequences
    #     # - vLLM thinker forward: thinker_result is the text hidden states tensor of shape [B, T, H]
    #     if isinstance(thinker_result, torch.Tensor):
    #         # vLLM path: full sequence. Compose talker embeddings by summing token embeds and thinker hidden states
    #         text_hidden_states = thinker_result.to(device)
    #         if text_hidden_states.dim() == 2:
    #             text_hidden_states = text_hidden_states.unsqueeze(0)
    #         fused_embeds = base_token_embeds + text_hidden_states

    #         # Resolve special tokens analogous to engine's logic
    #         def _get_talker_token(attr_name: str, config_name: str) -> Optional[int]:
    #             value = getattr(self.talker, attr_name, None)
    #             if value is None and hasattr(self, "talker_config"):
    #                 value = getattr(self.talker_config, config_name, None)
    #             return int(value) if value is not None else None

    #         text_bos_token = _get_talker_token("text_bos_token", "tts_text_start_token_id")
    #         if text_bos_token is not None:
    #             bos_embed = self.thinker.get_input_embeddings(
    #                 torch.tensor([[text_bos_token]], dtype=torch.long, device=device)
    #             )
    #         else:
    #             bos_embed = torch.zeros_like(fused_embeds[:, :1, :])

    #         # Approximate first reply step with last fused token
    #         first_reply_like = fused_embeds[:, -1:, :]
    #         talker_inputs_embeds = torch.cat([fused_embeds, bos_embed, first_reply_like], dim=1)

    #         talker_attention_mask = None
    #         if attention_mask is not None:
    #             talker_attention_mask = torch.cat(
    #                 [attention_mask, attention_mask.new_ones((1, 2))], dim=1
    #             ).to(device)

    #         return {
    #             "talker_inputs_embeds": talker_inputs_embeds,
    #             "talker_attention_mask": talker_attention_mask,
    #         }

    #     # HF path below mirrors the original generate() conversion
    #     embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(device)

    #     if thinker_kwargs.get("input_features") is not None:
    #         audio_ids_mask = input_ids == int(self.thinker_config.audio_token_index)
    #         audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
    #         audio_mask_tensor = torch.zeros(
    #             [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
    #             dtype=embeds_to_talker.dtype,
    #             device=device,
    #         )
    #         embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)

    #     if thinker_kwargs.get("pixel_values") is not None:
    #         image_ids_mask = input_ids == int(self.thinker_config.image_token_index)
    #         image_mask = image_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
    #         image_mask_tensor = torch.zeros(
    #             [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
    #             dtype=embeds_to_talker.dtype,
    #             device=device,
    #         )
    #         embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)

    #     if thinker_kwargs.get("pixel_values_videos") is not None:
    #         video_ids_mask = input_ids == int(self.thinker_config.video_token_index)
    #         video_mask = video_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
    #         video_mask_tensor = torch.zeros(
    #             [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
    #             dtype=embeds_to_talker.dtype,
    #             device=device,
    #         )
    #         embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

    #     processed_thinker_hidden = (
    #         (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
    #     ) + thinker_result.hidden_states[1:]

    #     thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(device)

    #     thinker_token_embeds = [
    #         token_hidden_states[0].to(device) for token_hidden_states in processed_thinker_hidden
    #     ]
    #     thinker_hidden_states = [
    #         token_hidden_states[-1].to(device) for token_hidden_states in processed_thinker_hidden
    #     ]

    #     def _get_talker_token(attr_name: str, config_name: str) -> Optional[int]:
    #         value = getattr(self.talker, attr_name, None)
    #         if value is None and hasattr(self, "talker_config"):
    #             value = getattr(self.talker_config, config_name, None)
    #         return int(value) if value is not None else None

    #     text_bos_token = _get_talker_token("text_bos_token", "tts_text_start_token_id")
    #     text_eos_token = _get_talker_token("text_eos_token", "tts_text_eos_token_id")
    #     text_pad_token = _get_talker_token("text_pad_token", "tts_text_pad_token_id")
    #     codec_mask_token = _get_talker_token("codec_mask_token", "tts_codec_mask_token_id")
    #     codec_pad_token = _get_talker_token("codec_pad_token", "tts_codec_pad_token_id")
    #     codec_bos_token = _get_talker_token("codec_bos_token", "tts_codec_start_token_id")

    #     if any(v is None for v in [text_bos_token, text_eos_token, text_pad_token, codec_mask_token, codec_pad_token, codec_bos_token]):
    #         raise ValueError("Missing required talker special tokens for thinker->talker conversion")

    #     talker_input_text_ids = torch.cat(
    #         [
    #             input_ids,
    #             torch.tensor([[text_bos_token]], dtype=torch.long, device=device),
    #             thinker_generate_ids[:, :1],
    #         ],
    #         dim=-1,
    #     )

    #     talker_input_ids = torch.cat(
    #         [
    #             torch.full_like(input_ids, fill_value=codec_mask_token),
    #             torch.tensor([[codec_pad_token]], dtype=torch.long, device=device),
    #             torch.tensor([[codec_bos_token]], dtype=torch.long, device=device),
    #         ],
    #         dim=1,
    #     )

    #     def thinker_embed_tokens(ids: torch.Tensor) -> torch.Tensor:
    #         return self.thinker.get_input_embeddings(ids)

    #     thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
    #     talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]

    #     talker_text_bos_token = torch.tensor([[text_bos_token]], dtype=torch.long, device=device)
    #     talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(device)

    #     talker_inputs_embeds = torch.cat(
    #         [
    #             talker_inputs_embeds,
    #             talker_text_bos_embed,
    #             thinker_reply_part[:, :1, :],
    #         ],
    #         dim=1,
    #     )

    #     eos_embedding = thinker_embed_tokens(
    #         torch.tensor([[text_eos_token]], dtype=torch.long, device=device)
    #     )
    #     pad_embedding = thinker_embed_tokens(
    #         torch.tensor([[text_pad_token]], dtype=torch.long, device=device)
    #     )

    #     thinker_reply_part = torch.cat(
    #         [
    #             thinker_reply_part[:, 1:, :],
    #             eos_embedding,
    #             pad_embedding,
    #         ],
    #         dim=1,
    #     )

    #     talker_attention_mask = None
    #     if attention_mask is not None:
    #         talker_attention_mask = torch.cat(
    #             [attention_mask, attention_mask.new_ones((1, 2))], dim=1
    #         ).to(device)

    #     return {
    #         "talker_input_ids": talker_input_ids,
    #         "talker_input_text_ids": talker_input_text_ids,
    #         "thinker_reply_part": thinker_reply_part,
    #         "talker_inputs_embeds": talker_inputs_embeds,
    #         "talker_attention_mask": talker_attention_mask,
    #     }
    # def thinker2talker(self, input_ids: torch.Tensor, thinker_result: torch.Tensor, device: torch.device, attention_mask: Optional[torch.Tensor] = None, **kwargs):
    #     # vLLM path with full sequence. Strictly follow lines 502-504:
    #     # tokens: [input_tokens] + [codec_pad_token] + [codec_bos_token]
    #     # embeddings: [input_embeds] + [text_bos_token] + [thinker_reply_part[0]]
    #     # thinker_reply_part: [thinker_reply_part[1:]] + [text_eos_token] + [text_pad_token]

    #     # 1) 选择输入 tokens（优先使用 self.prev_tokens）
    #     input_tokens = getattr(self, "prev_tokens", None)
    #     if input_tokens is None:
    #         input_tokens = input_ids
    #     input_tokens = input_tokens.to(device)  # shape (1, S)

    #     # 2) 取 talker 所需的特殊 token id（先读 self.talker 上的属性，fallback 到 self.talker_config）
    #     def _get_talker_token(attr_name: str, config_name: str) -> int:
    #         value = getattr(self.talker, attr_name, None)
    #         if value is None and hasattr(self, "talker_config"):
    #             value = getattr(self.talker_config, config_name, None)
    #         if value is None:
    #             raise ValueError(f"Missing token: {attr_name} / {config_name}")
    #         return int(value)

    #     text_bos_token = _get_talker_token("text_bos_token", "tts_text_start_token_id")
    #     text_eos_token = _get_talker_token("text_eos_token", "tts_text_eos_token_id")
    #     text_pad_token = _get_talker_token("text_pad_token", "tts_text_pad_token_id")
    #     codec_pad_token = _get_talker_token("codec_pad_token", "tts_codec_pad_token_id")
    #     codec_bos_token = _get_talker_token("codec_bos_token", "tts_codec_start_token_id")

    #     # 3) 计算三路输出
    #     # 3.1 tokens: [input_tokens] + [codec_pad_token] + [codec_bos_token]
    #     talker_input_ids = torch.cat([
    #         input_tokens,
    #         torch.tensor([[codec_pad_token]], dtype=torch.long, device=device),
    #         torch.tensor([[codec_bos_token]], dtype=torch.long, device=device),
    #     ], dim=1)

    #     # 3.2 embeddings: [input_embeds] + [text_bos_token] + [thinker_reply_part[0]]
    #     input_embeds = self.thinker.get_input_embeddings(input_tokens).to(device)  # (1, S, H)
    #     thinker_reply = thinker_result.to(device)                                  # (1, T, H)
    #     if thinker_reply.dim() == 2:
    #         thinker_reply = thinker_reply.unsqueeze(0)
    #     text_bos_embed = self.thinker.get_input_embeddings(
    #         torch.tensor([[text_bos_token]], dtype=torch.long, device=device)
    #     )
    #     first_reply = thinker_reply[:, :1, :]  # (1, 1, H)
    #     talker_inputs_embeds = torch.cat([input_embeds, text_bos_embed, first_reply], dim=1)

    #     # 3.3 thinker_reply_part: [thinker_reply_part[1:]] + [text_eos_token] + [text_pad_token]
    #     eos_embedding = self.thinker.get_input_embeddings(
    #         torch.tensor([[text_eos_token]], dtype=torch.long, device=device)
    #     )
    #     pad_embedding = self.thinker.get_input_embeddings(
    #         torch.tensor([[text_pad_token]], dtype=torch.long, device=device)
    #     )
    #     thinker_reply_part = torch.cat([thinker_reply[:, 1:, :], eos_embedding, pad_embedding], dim=1)

    #     # 4) attention_mask（若有）需要在时间维拼接两个 1
    #     talker_attention_mask = None
    #     if attention_mask is not None:
    #         talker_attention_mask = torch.cat(
    #             [attention_mask, attention_mask.new_ones((1, 2))], dim=1
    #         ).to(device)

    #     return {
    #         "talker_input_ids": talker_input_ids,
    #         "talker_inputs_embeds": talker_inputs_embeds,
    #         "thinker_reply_part": thinker_reply_part,
    #         "talker_attention_mask": talker_attention_mask,
    #     }

    def _convert_to_codec_tokens(self, talker_output: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """
        参考 HF：使用 talker 的 codec 头得到 logits，抑制 BOS，再贪心选取当前步的下一个 codec token。
        """
        with torch.inference_mode():
            logits = self.talker.compute_logits(talker_output, None)
            # # 依次尝试多种位置的 codec_head
            # codec_heads = []
            # if hasattr(self.talker, 'codec_head'):
            #     codec_heads.append(self.talker.codec_head)
            # if hasattr(self.talker, 'language_model') and hasattr(self.talker.language_model, 'codec_head'):
            #     codec_heads.append(self.talker.language_model.codec_head)
            # if hasattr(self.talker, 'language_model') and hasattr(self.talker.language_model, 'model') \
            #    and hasattr(self.talker.language_model.model, 'codec_head'):
            #     codec_heads.append(self.talker.language_model.model.codec_head)

            # for head in codec_heads:
            #     try:
            #         logits_tuple = head(talker_output)
            #         if isinstance(logits_tuple, tuple):
            #             logits = logits_tuple[0]
            #         else:
            #             logits = logits_tuple
            #         break
            #     except Exception:
            #         continue
            # 兜底：若不可用，返回空 codec
            if logits is None:
                return torch.zeros((talker_output.size(0), 0), dtype=torch.long, device=talker_output.device)

            # 仅抑制 codec_bos，与 HF generate 的 suppress_tokens 行为一致
            bos_id = None
            if hasattr(self, 'talker_config') and hasattr(self.talker_config, 'tts_codec_start_token_id'):
                bos_id = int(getattr(self.talker_config, 'tts_codec_start_token_id'))
            if bos_id is not None:
                logits[..., bos_id] = -1e9

            # 取最后一步位置的分布并贪心选取
            next_id = self.talker.sample(logits, sampling_metadata).sampled_token_ids
            return next_id.to(dtype=torch.long)

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
            cond = torch.zeros((1, self.token2wav_config.dit_config.enc_emb_dim), device=token2wav_dev, dtype=torch.float32)
        if ref_mel is None:
            ref_mel = torch.zeros((1, 300, self.token2wav_config.dit_config.mel_dim), device=token2wav_dev, dtype=torch.float32)
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
            
            t2w_loaded = self.token2wav.load_weights(token2wav_weights, os.path.join(self.vllm_config.model_config.model, "spk_dict.pt"))
            t2w_loaded = add_prefix_to_loaded_weights(t2w_loaded, 'token2wav')
            loaded_weights.update(t2w_loaded)
        
        return loaded_weights
