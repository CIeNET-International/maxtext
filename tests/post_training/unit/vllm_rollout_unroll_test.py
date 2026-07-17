# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Gemma scanned weights unrolling workaround."""

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from maxtext.integration.vllm.maxtext_vllm_rollout import (
    _create_model_converter,
    generate_maxtext_config_for_vllm,
    needs_maxtext_rollout,
    unroll_gemma_scanned_weights,
    unroll_scanned_weights,
)


class CreateModelConverterTest(unittest.TestCase):
  """Qwen MoE models need the torchax converter (QKV fusion, MoE expert fusion)
  on the stock-HF vLLM path; natively-served models (MaxTextForCausalLM) get
  None and rely on the scanned-weight unroll instead."""

  def _cfg(self):
    return SimpleNamespace(base_num_decoder_layers=48, rollout_tensor_parallelism=1)

  @pytest.mark.cpu_only
  def test_native_models_get_no_converter(self):
    for model in ("gemma3-4b", "gemma3-27b", "llama3-8b", "qwen3-8b"):
      self.assertIsNone(_create_model_converter(model, self._cfg(), mesh=None))

  @pytest.mark.cpu_only
  def test_qwen_moe_models_get_converter(self):
    conv = _create_model_converter("qwen3-30b-a3b", self._cfg(), mesh=None)
    self.assertIsNotNone(conv)
    self.assertEqual(conv.num_layers, 48)
    conv35 = _create_model_converter("qwen3.5-35b-a3b", self._cfg(), mesh=None)
    self.assertIsNotNone(conv35)


class HomogeneousScannedWeightsUnrollTest(unittest.TestCase):
  """Llama/Qwen-style scanned checkpoints stack params DIRECTLY under
  decoder/layers/<param> (no nested layers_N level), scan dim at axis 1.
  The unroll must insert integer layer indices for these too."""

  def _make_stacked(self, num_layers):
    # (feature_dim=2, scan_length=num_layers, 1); layer i holds value i.
    arr = np.zeros((2, num_layers, 1))
    for i in range(num_layers):
      arr[:, i, :] = i
    return arr

  @pytest.mark.cpu_only
  def test_unrolls_homogeneous_scan_with_base_prefix(self):
    arr_q = self._make_stacked(4)
    arr_wi = self._make_stacked(4) * 10
    pure_dict = {
        "base": {
            "decoder": {
                "layers": {
                    "self_attention": {"query": {"kernel": arr_q}},
                    "mlp": {"wi_0": {"kernel": arr_wi}},
                },
                "decoder_norm": {"scale": np.ones(3)},
            },
            "token_embedder": {"embedding": np.ones((4, 2))},
        }
    }
    unrolled = unroll_scanned_weights(MockWeights(pure_dict))

    layers = unrolled["base"]["decoder"]["layers"]
    for i in range(4):
      self.assertIn(i, layers)
      np.testing.assert_array_equal(layers[i]["self_attention"]["query"]["kernel"], np.full((2, 1), i))
      np.testing.assert_array_equal(layers[i]["mlp"]["wi_0"]["kernel"], np.full((2, 1), i * 10))
    # Non-layer entries pass through untouched.
    np.testing.assert_array_equal(unrolled["base"]["decoder"]["decoder_norm"]["scale"], np.ones(3))
    np.testing.assert_array_equal(unrolled["base"]["token_embedder"]["embedding"], np.ones((4, 2)))

  @pytest.mark.cpu_only
  def test_already_flat_string_indices_still_bypass(self):
    """Flat checkpoints with digit-string layer indices must stay untouched."""
    pure_dict = {
        "decoder": {
            "layers": {
                "0": {"attn": {"wq": np.ones((2, 3))}},
                "1": {"attn": {"wq": np.ones((2, 3))}},
            }
        }
    }
    weights = MockWeights(pure_dict)
    result = unroll_scanned_weights(weights)
    self.assertIs(result, weights)


class NeedsMaxtextRolloutTest(unittest.TestCase):
  """The MaxTextVllmRollout wrapper (weight unrolling) must engage for ANY scanned
  model on the native-vLLM path, not only Gemma: tunix blanks its key mappings for
  every model once `maxtext_config` is present, so scanned trainer weights never
  match the flat engine state without unrolling."""

  NATIVE_CFG = {"maxtext_config": {"model_name": "x"}}

  @pytest.mark.cpu_only
  def test_scanned_native_models_need_wrapper(self):
    for model in ("gemma3-4b", "llama3-8b", "qwen3-30b-a3b", "deepseek3-671b"):
      self.assertTrue(
          needs_maxtext_rollout(scan_layers=True, rollout_additional_config=self.NATIVE_CFG),
          msg=f"scanned native rollout must use the wrapper (model={model})",
      )

  @pytest.mark.cpu_only
  def test_non_native_path_keeps_plain_vllm(self):
    # Without maxtext_config, tunix keeps its real scanned->HF mappings.
    self.assertFalse(needs_maxtext_rollout(scan_layers=True, rollout_additional_config=None))
    self.assertFalse(needs_maxtext_rollout(scan_layers=True, rollout_additional_config={}))

  @pytest.mark.cpu_only
  def test_unscanned_native_path_keeps_plain_vllm(self):
    # Flat trainer keys already match the flat engine directly.
    self.assertFalse(needs_maxtext_rollout(scan_layers=False, rollout_additional_config=self.NATIVE_CFG))

  @pytest.mark.cpu_only
  def test_standalone_converter_flag_forces_wrapper(self):
    self.assertTrue(
        needs_maxtext_rollout(scan_layers=False, rollout_additional_config=None, use_standalone_converter=True)
    )


class GenerateMaxtextConfigForVllmTest(unittest.TestCase):
  """Verify the truncated config payload handed to the vLLM-side MaxText adapter."""

  def _mock_trainer_cfg(self):
    return SimpleNamespace(
        model_name="gemma3-4b",
        base_num_decoder_layers=34,
        base_emb_dim=2560,
        base_num_query_heads=8,
        base_num_kv_heads=4,
        base_mlp_dim=10240,
        vocab_size=262144,
        logits_via_embedding=True,
        decoder_block="gemma3",
        emb_dim=2560,
        num_query_heads=8,
        num_kv_heads=4,
        mlp_dim=10240,
        num_decoder_layers=34,
        attention="dot_product",
    )

  @pytest.mark.cpu_only
  def test_payload_does_not_clobber_engine_attention(self):
    """The trainer runs dot_product attention, but the rollout engine must keep the
    vllm_rpa backend from configs/inference/vllm.yml. The payload is merged as
    overrides on top of vllm.yml, so it must not carry the trainer's attention."""
    payload = generate_maxtext_config_for_vllm(self._mock_trainer_cfg())
    self.assertNotIn("attention", payload)

  @pytest.mark.cpu_only
  def test_payload_forces_flat_layers(self):
    """The rollout engine only supports unscanned layers."""
    payload = generate_maxtext_config_for_vllm(self._mock_trainer_cfg())
    self.assertIs(payload["scan_layers"], False)
    self.assertEqual(payload["model_name"], "gemma3-4b")


class MockWeights:
  """A mock weight container that implements to_pure_dict."""

  def __init__(self, pure_dict):
    self._pure_dict = pure_dict

  def to_pure_dict(self):
    return self._pure_dict


class GemmaScannedWeightsUnrollTest(unittest.TestCase):
  """Verify the correctness of the unroll_gemma_scanned_weights utility."""

  @pytest.mark.cpu_only
  def test_bypasses_non_pytree_weights(self):
    """If the weights object doesn't have `to_pure_dict`, it should be returned unchanged."""
    raw_weights = {"dummy": np.ones(5)}
    result = unroll_gemma_scanned_weights(raw_weights)
    self.assertIs(result, raw_weights)

  @pytest.mark.cpu_only
  def test_bypasses_non_scanned_checkpoints(self):
    """If the checkpoint is not scanned (no 'layers_0' inside 'decoder/layers/'), return unchanged."""
    pure_dict = {
        "decoder": {
            "layers": {
                "0": {"attn": {"wq": np.ones(10)}},
                "1": {"attn": {"wq": np.ones(10)}},
            }
        }
    }
    weights = MockWeights(pure_dict)
    result = unroll_gemma_scanned_weights(weights)
    self.assertIs(result, weights)

  @pytest.mark.cpu_only
  def test_correctly_unrolls_gemma_scanned_weights(self):
    """Verify that scanned layers are properly interleaved and mapped, and remainder layers are appended."""
    # Pattern length = 2 (layers_0 and layers_1)
    # Scan length = 3. In MaxText, param_scan_axis=1, so shape is (feature_dim, scan_length, ...)

    # We want an array where axis 1 has length 3. Let's make it (2, 3, 1)
    # For layers_0, values should be 0, 2, 4
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    # For layers_1, values should be 1, 3, 5
    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "layers": {
                "layers_0": {
                    "attn": {"wq": arr0},
                },
                "layers_1": {
                    "attn": {"wq": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "attn": {"wq": np.array([[6, 6]]).transpose()},  # shape (2, 1)
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_gemma_scanned_weights(weights)

    # Check unrolled structure. Layer indices must be INTEGERS so they match
    # the flat model's nnx.List state keys during tunix's direct key transfer.
    decoder_dict = unrolled["decoder"]

    for idx in range(7):
      self.assertIn(idx, decoder_dict["layers"])

    # Check that values are correctly sliced
    np.testing.assert_array_equal(decoder_dict["layers"][0]["attn"]["wq"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers"][1]["attn"]["wq"], np.array([[1], [1]]))
    np.testing.assert_array_equal(decoder_dict["layers"][2]["attn"]["wq"], np.array([[2], [2]]))
    np.testing.assert_array_equal(decoder_dict["layers"][3]["attn"]["wq"], np.array([[3], [3]]))
    np.testing.assert_array_equal(decoder_dict["layers"][4]["attn"]["wq"], np.array([[4], [4]]))
    np.testing.assert_array_equal(decoder_dict["layers"][5]["attn"]["wq"], np.array([[5], [5]]))
    np.testing.assert_array_equal(decoder_dict["layers"][6]["attn"]["wq"], np.array([[6], [6]]))

  @pytest.mark.cpu_only
  def test_correctly_unrolls_gemma3_gemma4_scanned_blocks(self):
    """Verify that scanned layers under scanned_blocks are properly interleaved and mapped."""
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "scanned_blocks": {
                "layers_0": {
                    "attn": {"wq": arr0},
                },
                "layers_1": {
                    "attn": {"wq": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "attn": {"wq": np.array([[6, 6]]).transpose()},
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_gemma_scanned_weights(weights)

    decoder_dict = unrolled["decoder"]
    self.assertIn(0, decoder_dict["layers"])
    self.assertIn(6, decoder_dict["layers"])
    np.testing.assert_array_equal(decoder_dict["layers"][0]["attn"]["wq"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers"][6]["attn"]["wq"], np.array([[6], [6]]))

  @pytest.mark.cpu_only
  def test_preserves_base_prefix(self):
    """Keys before 'decoder' (the TunixMaxTextAdapter 'base' attribute) must survive.

    tunix's transfer_state_directly unwraps src_state['base'] and silently
    discards everything outside it, so dropping the prefix loses all weights
    (the original gibberish bug)."""
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    pure_dict = {
        "base": {
            "decoder": {
                "layers": {
                    "layers_0": {
                        "attn": {"wq": arr0},
                    },
                },
            },
            "token_embedder": {"embedding": np.ones((4, 2))},
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_gemma_scanned_weights(weights)

    self.assertIn("base", unrolled)
    layers = unrolled["base"]["decoder"]["layers"]
    for idx in range(3):
      self.assertIn(idx, layers)
    np.testing.assert_array_equal(layers[0]["attn"]["wq"], np.array([[0], [0]]))
    np.testing.assert_array_equal(layers[2]["attn"]["wq"], np.array([[4], [4]]))
    # Passthrough (non-scanned) entries keep their full path too.
    np.testing.assert_array_equal(unrolled["base"]["token_embedder"]["embedding"], np.ones((4, 2)))
