#!/usr/bin/env python3
"""
End-to-end test example for Qwen2.5-Omni merged model using real checkpoints.

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

import argparse
import json
import os

import soundfile as sf
try:
    from huggingface_hub import snapshot_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False
import torch

from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


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


def get_model_instance(engine: AsyncLLMEngine):
    # Access underlying model instance from engine internals
    return engine.engine.llm_engine.model_executor.driver_worker.model_runner.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to merged model directory (will be created if downloading).')
    parser.add_argument('--hf-hub-id', default='Qwen/Qwen2.5-Omni-7B', help='Hugging Face repo id to download if needed.')
    parser.add_argument('--hf-revision', default=None, help='Optional HF revision (branch/tag/commit).')
    parser.add_argument('--prompt', required=True, help='Input text prompt.')
    parser.add_argument('--voice-type', default='default', help='Voice type, e.g., m02, f030, default.')
    parser.add_argument('--code2wav-dir', default=None, help='Path to code2wav folder (contains spk_dict.pt).')
    parser.add_argument('--dit-ckpt', default=None, help='Path to DiT checkpoint file (e.g., dit.pt).')
    parser.add_argument('--bigvgan-ckpt', default=None, help='Path to BigVGAN checkpoint file.')
    parser.add_argument('--output-wav', required=True, help='Output wav file path.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of identical prompts to batch together.')
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--max-model-len', type=int, default=32768)
    args = parser.parse_args()

    # 1) Download HF model if needed
    ensure_hf_model(args.model, args.hf_hub_id, args.hf_revision)
    # 2) Ensure config for merged model + code2wav
    ensure_config(args.model, args.code2wav_dir, args.dit_ckpt, args.bigvgan_ckpt)

    # Build engine
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Prepare input ids and positions using tokenizer
    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode='auto',
        trust_remote_code=True,
        revision=None,
        download_dir=None,
    )
    input_ids_list = tokenizer.encode(args.prompt)
    if not input_ids_list:
        raise ValueError('Tokenized prompt is empty.')

    device = engine.engine.device_config.device
    # Build a batch by repeating the same prompt to ensure inputs_embeds shape is (B, S, D)
    batch_size = max(1, int(args.batch_size))
    batch_input_ids = [input_ids_list for _ in range(batch_size)]
    input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)  # (B, S)
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    # Directly call model.forward to get both text and audio
    model = get_model_instance(engine)
    with torch.inference_mode():
        output = model.forward(
            input_ids=input_ids,
            positions=positions,
            generate_audio=True,
            voice_type=args.voice_type,
        )

    # Save audio
    audio = output.audio_tensor
    if audio is None or audio.numel() == 0:
        raise RuntimeError('No audio generated. Check code2wav checkpoints and voice type.')

    audio_cpu = audio.detach().cpu().numpy()
    # choose 24kHz as default; adjust if your BigVGAN is trained at other rates
    sf.write(args.output_wav, audio_cpu.squeeze(0), 24000)

    # Optional: also run text generation to see text output
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
    request_outputs = engine.generate(args.prompt, sampling)
    for ro in request_outputs:
        print('Generated text:', ro.outputs[0].text)

    # Cleanup
    engine.engine.llm_engine.shutdown()


if __name__ == '__main__':
    main()
