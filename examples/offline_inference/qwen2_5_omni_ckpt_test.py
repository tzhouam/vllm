#!/usr/bin/env python3
"""
End-to-end test example for Qwen2.5-Omni merged model using real checkpoints.
Modified for deterministic behavior - all randomness removed for debugging.

This script expects a merged checkpoint directory that includes a config.json
with architectures: ["Qwen2_5OmniMergedModel"], and thinker/talker weights.

It can also automatically download the official model repo from Hugging Face,
e.g. Qwen/Qwen2.5-Omni-7B, and prepare the directory for you.

It also requires Code2Wav checkpoints (DiT and BigVGAN), or a code2wav model
folder containing spk_dict.pt. You can provide either individual ckpt files or
a --code2wav-dir that contains spk_dict.pt and (optionally) model files.

Usage (Windows PowerShell):

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --model C:\\path\\to\\qwen2_5_omni_merged_ckpt \
  --prompt "你好，请介绍一下你自己。" \
  --voice-type m02 \
  --dit-ckpt C:\\path\\to\\code2wav\\dit.pt \
  --bigvgan-ckpt C:\\path\\to\\code2wav\\bigvgan.pt \
  --output-wav C:\\path\\to\\output.wav

Or if you have a code2wav folder:

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --model C:\\path\\to\\qwen2_5_omni_merged_ckpt \
  --prompt "请用中文介绍一下人工智能的发展历程。" \
  --voice-type m02 \
  --code2wav-dir C:\\path\\to\\code2wav_dir \
  --output-wav C:\\path\\to\\output.wav

Auto-download from Hugging Face (Qwen/Qwen2.5-Omni-7B):

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --hf-hub-id Qwen/Qwen2.5-Omni-7B \
  --model C:\\models\\Qwen2.5-Omni-7B \
  --prompt "请用中文介绍一下人工智能的发展历程。" \
  --voice-type default \
  --code2wav-dir C:\\models\\Qwen2.5-Omni-7B \
  --output-wav C:\\temp\\omni_out.wav
"""

import os
import random
import numpy as np
import torch

# =============================================================================
# DETERMINISTIC SETUP - MUST BE FIRST
# =============================================================================
SEED = 42  # Fixed seed for reproducibility

# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import copy
import json
import queue
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import librosa
import psutil
import requests
import resampy
import soundfile as sf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.multimodal.processing_omni import fetch_image, fetch_video
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

import asyncio
try:
    from huggingface_hub import snapshot_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

import os as _os_env_toggle
_os_env_toggle.environ["VLLM_USE_V1"] = "1"
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to merged model directory (will be created if downloading).')
parser.add_argument('--thinker-model', type=str, default=None)
parser.add_argument('--talker-model', type=str, default=None)
parser.add_argument('--code2wav-model', type=str, default=None)
parser.add_argument('--hf-hub-id', default='Qwen/Qwen2.5-Omni-7B', help='Hugging Face repo id to download if needed.')
parser.add_argument('--hf-revision', default=None, help='Optional HF revision (branch/tag/commit).')
parser.add_argument('--prompt', required=True, help='Input text prompt.')
parser.add_argument('--voice-type', default='default', help='Voice type, e.g., m02, f030, default.')
parser.add_argument('--code2wav-dir', default=None, help='Path to code2wav folder (contains spk_dict.pt).')
parser.add_argument('--dit-ckpt', default=None, help='Path to DiT checkpoint file (e.g., dit.pt).')
parser.add_argument('--bigvgan-ckpt', default=None, help='Path to BigVGAN checkpoint file.')
parser.add_argument('--output-wav', required=True, help='Output wav file path.')
parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
parser.add_argument('--max-model-len', type=int, default=32768)

parser.add_argument("--thinker-only", action="store_true")
parser.add_argument("--text-only", action="store_true")
parser.add_argument("--do-wave", action="store_true")
parser.add_argument('--prompt_type',
                    choices=[
                        'text', 'audio', 'audio-long', 'audio-long-chunks',
                        'audio-long-expand-chunks', 'image', 'video',
                        'video-frames', 'audio-in-video', 'audio-in-video-v2',
                        "audio-multi-round", "badcase-vl", "badcase-text",
                        "badcase-image-early-stop", "badcase-two-audios",
                        "badcase-two-videos", "badcase-multi-round",
                        "badcase-voice-type", "badcase-voice-type-v2",
                        "badcase-audio-tower-1", "badcase-audio-only"
                    ],
                    default='text')
parser.add_argument('--use-torchvision', action='store_true')
parser.add_argument('--tokenize', action='store_true')
args = parser.parse_args()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_hf_model(model_dir: str, hf_hub_id: str | None, revision: str | None):
    """Download model repo from HF into model_dir if needed."""
    if not hf_hub_id:
        return
    if not _HF_AVAILABLE:
        raise RuntimeError("huggingface_hub is not installed. Please `pip install -U huggingface_hub`." )
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.isdir(model_dir) and os.path.exists(cfg_path):
        return  # already prepared
    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        repo_id=hf_hub_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        revision=revision,
        ignore_patterns=['.git/*'],
    )

def ensure_config(model_dir: str, code2wav_dir: str | None, dit_ckpt: str | None, bigvgan_ckpt: str | None):
    """Ensure the merged model's config.json has code2wav_config fields set.

    This function updates/creates config.json in-place to include:
      - architectures: ["Qwen2_5OmniMergedModel"]
      - code2wav_config: { dit_checkpoint, bigvgan_checkpoint, model_path?, frequency }
    It does not modify thinker/talker sub-configs; those should already match your weights.
    """
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {}

    archs = cfg.get('architectures') or []
    if 'Qwen2_5OmniMergedModel' not in archs:
        cfg['architectures'] = ['Qwen2_5OmniMergedModel']

    code2wav_cfg = cfg.get('code2wav_config') or {}
    if code2wav_dir:
        code2wav_cfg['model_path'] = code2wav_dir
    if dit_ckpt:
        code2wav_cfg['dit_checkpoint'] = dit_ckpt
    if bigvgan_ckpt:
        code2wav_cfg['bigvgan_checkpoint'] = bigvgan_ckpt
    code2wav_cfg.setdefault('frequency', '50hz')
    cfg['code2wav_config'] = code2wav_cfg

    os.makedirs(model_dir, exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def get_model_instance(engine):
    # V1 AsyncLLM 不暴露底层模型句柄；如需底层信息，请通过 EngineCore/collective_rpc 查询
    return None

def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data,
                                      sr_orig=original_sample_rate,
                                      sr_new=16000)
    return data_resampled

def fetch_and_read_video(video_url: str, fps=2):
    import torchvision.io

    def read_video_with_torchvision(video_file_name: str):
        video, audio, info = torchvision.io.read_video(
            video_file_name,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )

        total_frames, video_fps = video.size(0), info["video_fps"]
        total_duration = round(total_frames / video_fps, 3)
        nframes = int(total_frames / video_fps * fps)

        frame_timestamps = total_duration * torch.arange(1,
                                                         nframes + 1) / nframes
        grid_timestamps = frame_timestamps[::2]
        second_per_grid = grid_timestamps[1] - grid_timestamps[0]

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        video_height, video_width = video.shape[2:]
        video = video[idx]

        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid.item()]
        else:
            return video

    def read_video_with_transformers(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {'video': video_file_name})
        if total_duration is None and nframes is None:
            nframes = len(video)
            total_duration = 0.5 * nframes
            second_per_grid = 1.0
        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid]
        else:
            return video

    def read_video(video_file_name: str):
        if args.use_torchvision:
            return read_video_with_torchvision(video_file_name)
        else:
            return read_video_with_transformers(video_file_name)

    if isinstance(video_url, str) and video_url.startswith("http"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
            resp = requests.get(video_url)
            assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)

# =============================================================================
# PROMPT CREATION FUNCTIONS
# =============================================================================
def make_inputs_qwen2_omni(
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    use_audio_in_video: Optional[bool] = False,
    tokenize: bool = args.tokenize,
) -> Union[TokensPrompt, TextPrompt]:
    
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        config = AutoConfig.from_pretrained(args.model)
        if 'Qwen2_5OmniModel' in config.architectures:
            args.legacy_omni_video = False
        else:
            args.legacy_omni_video = True
    except:
        args.legacy_omni_video = True

    audios, images, videos = [], [], []
    for message in messages:
        if not isinstance(message['content'], list):
            message['content'] = [{
                'type': 'text',
                'text': message['content'],
            }]
        index, num_contents = 0, len(message['content'])
        while index < num_contents:
            ele = message['content'][index]
            if 'type' not in ele:
                if 'text' in ele:
                    ele['type'] = 'text'
                elif 'audio' in ele:
                    ele['type'] = 'audio'
                elif 'audio_url' in ele:
                    ele['type'] = 'audio_url'
                elif 'image' in ele:
                    ele['type'] = 'image'
                elif 'image_url' in ele:
                    ele['type'] = 'image_url'
                elif 'video' in ele:
                    ele['type'] = 'video'
                elif 'video_url' in ele:
                    ele['type'] = 'video_url'
                else:
                    raise ValueError(f'Unknown ele: {ele}')

            if ele['type'] == 'audio' or ele['type'] == 'audio_url':
                if 'audio_url' in ele:
                    audio_key = 'audio_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_audio_file:
                        temp_audio_file.write(urlopen(ele[audio_key]).read())
                        temp_audio_file_path = temp_audio_file.name
                        audios.append(
                            resample_wav_to_16khz(temp_audio_file_path))
                        ele['audio'] = temp_audio_file_path
                elif 'audio' in ele:
                    audio_key = 'audio'
                    audios.append(resample_wav_to_16khz(ele[audio_key]))
                else:
                    raise ValueError(f'Unknown ele {ele}')
            elif use_audio_in_video and (ele['type'] == 'video'
                                         or ele['type'] == 'video_url'):
                # use video as audio as well
                if 'video_url' in ele:
                    audio_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele[audio_key]).read())
                        temp_video_file_path = temp_video_file.name
                        ele[audio_key] = temp_video_file_path
                        audios.append(
                            librosa.load(temp_video_file_path, sr=16000)[0])
                        videos.append(
                            fetch_and_read_video(temp_video_file_path))
                        ele['video'] = temp_video_file_path
                elif 'video' in ele:
                    audio_key = 'video'
                    audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                    videos.append(fetch_and_read_video(audio_key))
                else:
                    raise ValueError("Unknown ele {}".format(ele))
                # insert a audio after the video
                message['content'].insert(index + 1, {
                    "type": "audio",
                    "audio": ele[audio_key],
                })
                # no need to load the added audio again
                index += 1
            elif ele['type'] == 'video' or ele['type'] == 'video_url':
                if 'video_url' in ele:
                    video_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele['video_url']).read())
                        temp_video_file_path = temp_video_file.name
                        videos.append(fetch_and_read_video(temp_video_file))
                        ele['video'] = temp_video_file_path
                else:
                    video_key = 'video'
                    videos.append(fetch_and_read_video(ele[video_key]))
            elif ele['type'] == 'image' or ele['type'] == 'image_url':
                images.append(fetch_image(ele))

            # move to the next content
            index += 1

    prompt = processor.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        add_vision_id=True,
    )

    audios = audios if len(audios) > 0 else None
    images = images if len(images) > 0 else None
    videos = videos if len(videos) > 0 else None

    multi_modal_data = {}
    if audios:
        multi_modal_data["audio"] = audios
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    if isinstance(prompt, list) and isinstance(prompt[0], (list, str)):
        prompt = prompt[0]

    if tokenize:
        return TokensPrompt(
            prompt_token_ids=prompt,
            multi_modal_data=multi_modal_data,
        )
    else:
        return TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )

def get_system_prompt():
    
    return {
        'role':
        'system',
        'content': [{
            'type':
            'text',
            'text':
            'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
        }]
    }

def make_text_prompt():
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": args.prompt
                },
            ]
        },
    ]

    prompt = make_inputs_qwen2_omni(messages, )
    return prompt

def make_audio_in_video_v2_prompt():
    messages = [
        {
            'role':
            'system',
            'content': [{
                'type':
                'text',
                'text':
                args.prompt
            }]
        },
        # {
        #     "role":
        #     "user",
        #     "content": [
        #         {
        #             "type":
        #             "video_url",
        #             "video_url":
        #             "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw_small.mp4"
        #         },
        #     ]
        # },
    ]
    prompt = make_inputs_qwen2_omni(
        messages,
        use_audio_in_video=True,
    )
    return prompt

def make_omni_prompt() -> Union[TokensPrompt, List[TokensPrompt]]:
    if args.prompt_type == 'text':
        prompt = make_text_prompt()
    elif args.prompt_type == 'audio-in-video-v2':
        prompt = make_audio_in_video_v2_prompt()
    else:
        raise ValueError(f'Unsupported prompt type: {args.prompt_type}')
    return prompt

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print(f"Running with deterministic seed: {SEED}")
    print(f"PyTorch deterministic mode: {torch.backends.cudnn.deterministic}")
    print(f"PyTorch benchmark mode: {torch.backends.cudnn.benchmark}")
    
    # 1) Download HF model if needed
    ensure_hf_model(args.model, args.hf_hub_id, args.hf_revision)
    
    # 2) Ensure config for merged model + code2wav
    ensure_config(args.model, args.code2wav_dir, args.dit_ckpt, args.bigvgan_ckpt)
    
    # 3) Create prompt
    prompt = make_omni_prompt()
    
    # 4) Build engine with deterministic settings
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        seed=SEED,  # Set deterministic seed
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    # 5) Use deterministic sampling parameters
    # sampling = SamplingParams(
    #     temperature=0.0,    # Deterministic - no randomness
    #     top_p=1.0,          # Disable nucleus sampling
    #     top_k=-1,           # Disable top-k sampling
    #     max_tokens=128,
    #     seed=SEED,          # Fixed seed for sampling
    # )

    sampling = SamplingParams(
        temperature=0.9,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=2048,
        detokenize=False,
        seed=SEED,
    )

    print(f"Sampling parameters: temperature={sampling.temperature}, top_p={sampling.top_p}, seed={sampling.seed}")

    async def run_generation():
        final_output = None
        request_id = f"req-omni-deterministic-{SEED}"
        
        async for ro in engine.generate(prompt, sampling, request_id):
            final_output = ro
        return final_output

    # 6) Run generation
    final = asyncio.run(run_generation())
    
    if final is None or not final.outputs:
        raise RuntimeError('No text generated from engine.')
    
    print('Generated text:', final.outputs[0].text)
    print(f"Request ID: {final.request_id}")
    print(f"Finished reason: {final.outputs[0].finish_reason}")

if __name__ == '__main__':
    main()