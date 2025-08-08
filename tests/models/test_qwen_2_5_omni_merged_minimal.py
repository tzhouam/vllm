import types
from unittest.mock import Mock, patch

import torch
import pytest


def _make_fake_vllm_config():
    # minimal structure to satisfy model __init__ attribute access
    thinker_cfg = types.SimpleNamespace()
    talker_cfg = types.SimpleNamespace(
        # special codec token ids used in _convert_to_codec_tokens masking
        tts_codec_start_token_id=1,
        tts_codec_end_token_id=2,
        tts_codec_pad_token_id=0,
        embedding_size=4096,
        hidden_size=4096,
    )
    code2wav_cfg = None  # lazy init path won't trigger
    hf_config = types.SimpleNamespace(
        thinker_config=thinker_cfg,
        talker_config=talker_cfg,
        code2wav_config=code2wav_cfg,
    )
    model_config = types.SimpleNamespace(hf_config=hf_config, multimodal_config=None)
    vllm_config = types.SimpleNamespace(model_config=model_config, quant_config=None)
    return vllm_config


class _StubCode2Wav:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def __call__(self, cond, ref_mel, codec):
        # return a fake waveform [B, N] float32
        b = 1 if cond is None else cond.shape[0]
        return torch.randn(b, 16000, dtype=torch.float32, device=self.device)


@pytest.fixture()
def mocks_for_init():
    # thinker mock: returns hidden states [B, T, H]
    mock_thinker = Mock()
    def thinker_forward(*args, **kwargs):
        b = kwargs.get("input_ids").shape[0] if kwargs.get("input_ids") is not None else 1
        t = kwargs.get("input_ids").shape[1] if kwargs.get("input_ids") is not None else 8
        return torch.randn(b, t, 4096)
    mock_thinker.side_effect = thinker_forward
    mock_thinker.make_empty_intermediate_tensors = lambda: None

    # talker mock: has language_model.lm_head returning logits
    mock_talker = Mock()
    mock_lm = types.SimpleNamespace()
    def lm_head(x):
        # x: [B, T=1, H] -> logits [B, T=1, V]
        b, t, _ = x.shape
        return torch.randn(b, t, 128)
    mock_lm.lm_head = lm_head
    mock_talker.language_model = mock_lm

    # init_vllm_registered_model should return thinker then talker
    return [mock_thinker, mock_talker]


def test_forward_text_only(mocks_for_init):
    from vllm.model_executor.models.qwen_2_5_omni import Qwen2_5OmniForConditionalGeneration

    with patch('vllm.model_executor.models.qwen_2_5_omni.init_vllm_registered_model') as mock_init:
        mock_init.side_effect = mocks_for_init

        model = Qwen2_5OmniForConditionalGeneration(
            vllm_config=_make_fake_vllm_config(),
            prefix="",
        )

        input_ids = torch.randint(0, 100, (1, 8))
        positions = torch.arange(8).unsqueeze(0)

        out = model.forward(
            input_ids=input_ids,
            positions=positions,
            generate_audio=False,
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape[:2] == (1, 8)


def test_forward_with_audio(mocks_for_init):
    from vllm.model_executor.models.qwen_2_5_omni import Qwen2_5OmniForConditionalGeneration

    with patch('vllm.model_executor.models.qwen_2_5_omni.init_vllm_registered_model') as mock_init:
        mock_init.side_effect = mocks_for_init

        model = Qwen2_5OmniForConditionalGeneration(
            vllm_config=_make_fake_vllm_config(),
            prefix="",
        )

        # inject stub code2wav so that _codec_to_audio works without checkpoints
        model.code2wav_model = _StubCode2Wav(device="cpu")

        input_ids = torch.randint(0, 100, (1, 8))
        positions = torch.arange(8).unsqueeze(0)

        out = model.forward(
            input_ids=input_ids,
            positions=positions,
            generate_audio=True,
            voice_type="default",
        )
        # OmniOutput NamedTuple
        assert hasattr(out, 'text_hidden_states')
        assert hasattr(out, 'audio_tensor')
        assert isinstance(out.audio_tensor, torch.Tensor)
        assert out.audio_tensor.ndim == 2
        assert out.audio_tensor.shape[0] == 1
