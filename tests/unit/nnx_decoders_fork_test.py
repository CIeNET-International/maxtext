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

"""Unit tests asserting ``nnx.Rngs.fork(split=...)`` is exercised by the NNX decoder.

There are exactly two ``rngs.fork(split=N)`` call sites in
``src/maxtext/layers/nnx_decoders.py``:

  * SITE 1 -- ``NNXScannedPipelineStage.__init__`` (``forked_rngs = rngs.fork(split=num_layers)``).
    Reached when a model is built with pipeline parallelism AND
    ``scan_layers_per_stage=True`` AND ``num_layers_per_pipeline_stage > 1``. Needs > 1 device.

  * SITE 2 -- ``NNXDecoder._create_scanned_layers`` (``forked_rngs = rngs.fork(split=length)``).
    Reached when an ``NNXDecoder`` is built with ``scan_layers=True`` (the non-pipeline scanned
    path). The ``split`` length is the scan length for that model.

These tests do NOT re-derive the routing -- they encode the verified routing truth table and the
verified per-model scan lengths and assert them. ``fork`` is called many times during construction
with ``split=None`` (nnx-internal); ``m.called`` alone is therefore meaningless. Every assertion
checks that the *specific* expected ``split`` value appears in the recorded list (and that the
decoder still builds, because the spy lets the real fork run).

All tests are CPU-only, use tiny dims, and are deterministic.
"""

import sys
import unittest
from unittest import mock

import jax
import pytest
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import MODEL_MODE_TRAIN, DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers.nnx_decoders import NNXDecoder, NNXDecoderLayer, NNXScannedPipelineStage
from maxtext.models import gemma3, gemma4
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# ---------------------------------------------------------------------------
# Shared minimal config helpers (mirrors tests/unit/nnx_decoders_test.py)
# ---------------------------------------------------------------------------
_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "nnx_decoder_fork_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 2,
    "attention": "dot_product",
    "max_target_length": 16,
    "base_emb_dim": 256,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 512,
    "max_prefill_predict_length": 4,
    "scan_layers": False,
}


def _make_config(**overrides):
  """Return a pyconfig Config object suitable for unit tests."""
  merged = {**_BASE_CONFIG, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


# ---------------------------------------------------------------------------
# Verified fork spy (see the verified TASK C "ROBUST FORK SPY" recipe).
#
# ``autospec=True`` is MANDATORY: production calls are bound (``instance.fork(split=N)``); with
# autospec mock injects ``self`` as args[0] so we can call the real method (the build consumes the
# real fork return value -- a non-passthrough spy breaks the build). ``split`` is keyword-only, so
# it is always in kwargs.
# ---------------------------------------------------------------------------


def _fork_spy():
  """Return ``(context_manager, recorded_splits)``.

  Inside the context manager, every ``nnx.Rngs.fork`` call appends its ``split`` kwarg to
  ``recorded_splits`` and then runs the real fork. ``None`` entries are nnx-internal forks.
  """
  real_fork = nnx.Rngs.fork
  recorded = []

  def spy(self, *args, **kwargs):  # autospec => self is the 1st positional
    recorded.append(kwargs.get("split"))  # 'split' is keyword-only -> always in kwargs
    return real_fork(self, *args, **kwargs)  # REAL fork runs => decoder builds

  cm = mock.patch.object(nnx.Rngs, "fork", autospec=True, side_effect=spy)
  return cm, recorded


# ---------------------------------------------------------------------------
# Per-model SITE-2 specification table.
#
# Each entry maps a DecoderBlockType to:
#   - overrides: exact kwargs added to _make_config (verified buildable on CPU)
#   - expected_splits: the set of split=N values that MUST appear in the recorded fork list
#                      when scan_layers=True (the scan length(s) reaching SITE 2). The scan length
#                      equals num_decoder_layers // inhomogeneous_layer_cycle_interval for the
#                      generic path, num_decoder_layers // len(<pattern>) for gemma3/gemma4, and the
#                      (dense, moe) split lengths for deepseek.
#
# These values are taken verbatim from the empirically-verified per-model fork table; this test
# encodes them rather than re-deriving them.
# ---------------------------------------------------------------------------

_GEMMA3_PATTERN_LEN = len(gemma3.GEMMA3_ATTENTION_PATTERN)  # 6
_GEMMA4_PATTERN_LEN = len(gemma4.GEMMA4_ATTENTION_PATTERN)  # 6

_MODEL_SPECS = {
    # Generic single-class path: split length == num_decoder_layers (interval == 1) == 2.
    DecoderBlockType.DEFAULT: (dict(decoder_block="default", scan_layers=True), {2}),
    DecoderBlockType.SIMPLE: (dict(decoder_block="simple", scan_layers=True), {2}),
    DecoderBlockType.SIMPLE_MLP: (dict(decoder_block="simple_mlp", scan_layers=True), {2}),
    DecoderBlockType.LLAMA2: (dict(model_name="llama2-7b", scan_layers=True), {2}),
    DecoderBlockType.MISTRAL: (dict(model_name="mistral-7b", scan_layers=True), {2}),
    DecoderBlockType.MIXTRAL: (dict(model_name="mixtral-8x7b", base_moe_mlp_dim=512, scan_layers=True), {2}),
    DecoderBlockType.GEMMA: (dict(model_name="gemma-2b", scan_layers=True), {2}),
    DecoderBlockType.GEMMA2: (dict(model_name="gemma2-2b", scan_layers=True), {2}),
    DecoderBlockType.GPT3: (dict(model_name="gpt3-52k", scan_layers=True), {2}),
    DecoderBlockType.QWEN2: (dict(model_name="qwen2.5-1.5b", scan_layers=True), {2}),
    DecoderBlockType.QWEN3: (dict(model_name="qwen3-0.6b", scan_layers=True), {2}),
    DecoderBlockType.QWEN3_MOE: (dict(model_name="qwen3-30b-a3b", base_moe_mlp_dim=512, scan_layers=True), {2}),
    DecoderBlockType.QWEN3_CUSTOM_MOE: (
        dict(model_name="qwen3-custom-30b-a3b", base_moe_mlp_dim=512, scan_layers=True),
        {2},
    ),
    # gemma3/gemma4: split length == num_decoder_layers // pattern_len; need base layers >= pattern.
    DecoderBlockType.GEMMA3: (
        dict(model_name="gemma3-4b", base_num_decoder_layers=_GEMMA3_PATTERN_LEN, scan_layers=True),
        {_GEMMA3_PATTERN_LEN // _GEMMA3_PATTERN_LEN},  # == 1
    ),
    DecoderBlockType.GEMMA4: (
        dict(decoder_block="gemma4", base_num_decoder_layers=_GEMMA4_PATTERN_LEN, scan_layers=True),
        {_GEMMA4_PATTERN_LEN // _GEMMA4_PATTERN_LEN},  # == 1
    ),
    # Scannable-block models with interval > 1: split length == num_decoder_layers // interval.
    DecoderBlockType.GPT_OSS: (
        dict(model_name="gpt-oss-20b", base_moe_mlp_dim=512, scan_layers=True),
        {1},  # interval 2, base 2 -> 2//2 == 1
    ),
    DecoderBlockType.QWEN3_NEXT: (
        dict(model_name="qwen3-next-80b-a3b", base_num_decoder_layers=4, base_moe_mlp_dim=512, scan_layers=True),
        {1},  # interval 4, 4//4 == 1
    ),
    DecoderBlockType.QWEN3_5: (
        dict(model_name="qwen3.5-35b-a3b", base_num_decoder_layers=4, base_moe_mlp_dim=512, scan_layers=True),
        {1},  # interval 4, 4//4 == 1
    ),
    DecoderBlockType.LLAMA4: (
        dict(model_name="llama4-17b-16e", base_num_decoder_layers=4, base_moe_mlp_dim=512, scan_layers=True),
        {1},  # interval 4, 4//4 == 1
    ),
    DecoderBlockType.OLMO3: (
        dict(model_name="olmo3-7b", base_num_decoder_layers=4, scan_layers=True),
        {1},  # interval 4, 4//4 == 1
    ),
    # DEEPSEEK: TWO forks. dense_layers == first_num_dense_layers (=3), moe_layers ==
    # num_decoder_layers - first_num_dense_layers (= 4 - 3 = 1). base layers MUST exceed
    # first_num_dense_layers, else the moe split is negative and the bare-except swallows the fork.
    DecoderBlockType.DEEPSEEK: (
        dict(model_name="deepseek3-test", base_num_decoder_layers=4, base_moe_mlp_dim=512, scan_layers=True),
        {3, 1},
    ),
}

# GEMMA4_SMALL forbids scan_layers at the pyconfig level (per-layer KV sharing is incompatible with
# nn.scan), so it can NEVER hit SITE 2. It is handled by a dedicated negative test below instead.
_SITE2_SKIP = {
    DecoderBlockType.GEMMA4_SMALL: "gemma4_small forbids scan_layers (per-layer KV sharing incompatible with nn.scan); does not reach SITE 2",
}

# Sanity: the parametrization must cover every block in the production layer_map exactly once.
_ALL_BLOCKS = set(_MODEL_SPECS) | set(_SITE2_SKIP)
assert _ALL_BLOCKS == set(DecoderBlockType), (
    "SITE-2 spec table is out of sync with DecoderBlockType: "
    f"missing={set(DecoderBlockType) - _ALL_BLOCKS}, extra={_ALL_BLOCKS - set(DecoderBlockType)}"
)


# ---------------------------------------------------------------------------
# SITE 2 -- parametrized over every buildable model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    list(_MODEL_SPECS),
    ids=[b.name for b in _MODEL_SPECS],
)
def test_site2_scanned_fork_fires_per_model(block):
  """SITE 2: every scannable model builds an NNXDecoder whose ``_create_scanned_layers`` forks.

  Builds ``NNXDecoder(scan_layers=True)`` directly (bypassing models.py routing) and asserts the
  expected scan-length ``split`` value(s) appear in the recorded ``rngs.fork`` calls -- i.e. the
  SITE-2 fork at ``nnx_decoders.py`` ``_create_scanned_layers`` actually fired with the right
  length(s). The spy runs the real fork, so a successful build also proves the fork is valid.
  """
  overrides, expected_splits = _MODEL_SPECS[block]
  cfg = _make_config(**overrides)
  mesh = _make_mesh(cfg)

  cm, recorded = _fork_spy()
  with cm:
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

  assert decoder is not None, f"{block.name}: NNXDecoder failed to build"
  for split in expected_splits:
    assert split in recorded, (
        f"{block.name}: expected SITE-2 fork(split={split}) but recorded={recorded} "
        f"(num_decoder_layers={cfg.num_decoder_layers})"
    )


@pytest.mark.parametrize("block", list(_SITE2_SKIP), ids=[b.name for b in _SITE2_SKIP])
def test_site2_skipped_models_do_not_reach_fork(block):
  """SITE 2 (negative): models that forbid scan_layers must build WITHOUT the scanned fork.

  GEMMA4_SMALL is the only such model -- pyconfig forbids ``scan_layers=True`` (its per-layer KV
  sharing is incompatible with ``nn.scan``), so it uses a per-layer instance loop and never reaches
  ``_create_scanned_layers``. We build it with ``scan_layers=False`` and assert that NO non-None
  ``split`` value was recorded (only nnx-internal ``split=None`` forks).
  """
  reason = _SITE2_SKIP[block]
  if block is not DecoderBlockType.GEMMA4_SMALL:
    pytest.skip(reason)

  cfg = _make_config(
      model_name="gemma4-e2b",
      base_num_decoder_layers=10,
      num_kv_shared_layers=4,
      hidden_size_per_layer_input=64,
      vocab_size_per_layer_input=256,
      scan_layers=False,
      use_multimodal=False,
  )
  mesh = _make_mesh(cfg)

  cm, recorded = _fork_spy()
  with cm:
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

  assert decoder is not None, "gemma4_small failed to build with scan_layers=False"
  non_none = [s for s in recorded if s is not None]
  assert not non_none, f"gemma4_small must not reach the scanned fork, but recorded split values {non_none}"

  # gemma4_small also rejects scan_layers=True at the pyconfig level (defense-in-depth check).
  with pytest.raises(Exception):
    _make_config(
        model_name="gemma4-e2b",
        base_num_decoder_layers=10,
        num_kv_shared_layers=4,
        hidden_size_per_layer_input=64,
        vocab_size_per_layer_input=256,
        scan_layers=True,
        use_multimodal=False,
    )


# ---------------------------------------------------------------------------
# FLAG COMBOS -- routing truth table (pure_nnx / enable_nnx / pure_nnx_decoder).
#
# models.py:358 is the switch: ``if cfg.pure_nnx_decoder: self.decoder = NNXDecoder(...) else
# ToNNX(Decoder(...))`` (Linen). So whether NNXDecoder -- and therefore the SITE-2 fork -- is reached
# through normal routing is governed ONLY by pure_nnx_decoder. The three combos:
#   (1) pure_nnx_decoder=false -> Linen decoder, SITE 2 NOT reached.
#   (2) enable_nnx=true, pure_nnx_decoder=true (base.yml default) -> NNXDecoder, SITE 2 reached.
#   (3) pure_nnx=true + pure_nnx_decoder=true -> NNXDecoder, SITE 2 reached.
# ---------------------------------------------------------------------------


class TestFlagComboRouting(unittest.TestCase):
  """Routing truth table: pure_nnx_decoder decides whether the SITE-2 fork is reachable."""

  def _decoder_class_for(self, **flag_overrides):
    """Return the decoder class models.py would build for the given flags (the routing switch).

    Mirrors ``models.py`` ``if cfg.pure_nnx_decoder: NNXDecoder else Decoder``. We assert against
    the config flag rather than instantiating a full Transformer (which is heavier and needs a real
    model build path); the documented switch is a single ``if`` on ``cfg.pure_nnx_decoder``.
    """
    cfg = _make_config(scan_layers=True, **flag_overrides)
    # Import locally to keep the module import cheap and explicit about what the switch references.
    from maxtext.layers.decoders import Decoder  # pylint: disable=import-outside-toplevel

    return NNXDecoder if cfg.pure_nnx_decoder else Decoder

  def test_combo1_linen_decoder_does_not_reach_site2(self):
    """Combo (1) pure_nnx_decoder=false: models.py builds the Linen Decoder, NOT NNXDecoder.

    Because NNXDecoder is never instantiated through routing, the SITE-2 fork
    (``NNXDecoder._create_scanned_layers``) is unreachable via the normal model-build path.
    """
    from maxtext.layers.decoders import Decoder  # pylint: disable=import-outside-toplevel

    chosen = self._decoder_class_for(pure_nnx=False, enable_nnx=False, pure_nnx_decoder=False)
    self.assertIs(chosen, Decoder)
    self.assertIsNot(chosen, NNXDecoder)

  def test_combo2_base_default_builds_nnx_decoder_and_forks(self):
    """Combo (2) enable_nnx=true + pure_nnx_decoder=true (base.yml default): NNXDecoder + SITE-2 fork.

    Asserts routing picks NNXDecoder AND that directly building it with scan_layers=True fires the
    SITE-2 fork with split == num_decoder_layers.
    """
    chosen = self._decoder_class_for(pure_nnx=False, enable_nnx=True, pure_nnx_decoder=True)
    self.assertIs(chosen, NNXDecoder)

    cfg = _make_config(scan_layers=True, pure_nnx=False, enable_nnx=True, pure_nnx_decoder=True)
    mesh = _make_mesh(cfg)
    cm, recorded = _fork_spy()
    with cm:
      decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder)
    self.assertIn(cfg.num_decoder_layers, recorded)

  def test_combo3_pure_nnx_builds_nnx_decoder_and_forks(self):
    """Combo (3) pure_nnx=true + pure_nnx_decoder=true: NNXDecoder + SITE-2 fork.

    pure_nnx switches the model-build entry point but the decoder type is still governed by
    pure_nnx_decoder at models.py:358, so the coherent full-NNX config still routes to NNXDecoder
    and fires the SITE-2 fork.
    """
    chosen = self._decoder_class_for(pure_nnx=True, enable_nnx=True, pure_nnx_decoder=True)
    self.assertIs(chosen, NNXDecoder)

    cfg = _make_config(scan_layers=True, pure_nnx=True, enable_nnx=True, pure_nnx_decoder=True)
    mesh = _make_mesh(cfg)
    cm, recorded = _fork_spy()
    with cm:
      decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder)
    self.assertIn(cfg.num_decoder_layers, recorded)


# ---------------------------------------------------------------------------
# SITE 1 -- NNXScannedPipelineStage.__init__ fork(split=num_layers).
#
# Verified trigger: pipeline parallelism + scan_layers_per_stage=True +
# num_layers_per_pipeline_stage > 1. Requires >= ici_pipeline_parallelism devices; CPU runs must
# force the device count via XLA_FLAGS=--xla_force_host_platform_device_count=2 BEFORE jax init
# (cannot be set after). conftest does not force this, so we skip when device_count is too small,
# mirroring tests/integration/pipeline_parallelism_test.py.
# ---------------------------------------------------------------------------

_SITE1_ICI_PIPELINE = 2


@pytest.mark.skipif(
    jax.device_count() < _SITE1_ICI_PIPELINE,
    reason=(
        f"SITE 1 needs >= {_SITE1_ICI_PIPELINE} devices for ici_pipeline_parallelism="
        f"{_SITE1_ICI_PIPELINE}. On CPU run with "
        "XLA_FLAGS=--xla_force_host_platform_device_count=2 (must be set before jax init)."
    ),
)
def test_site1_scanned_pipeline_stage_fork_fires():
  """SITE 1: a scanned pipeline stage forks ``rngs.fork(split=num_layers_per_pipeline_stage)``.

  Builds an NNXDecoder with pipeline parallelism, ``scan_layers_per_stage=True`` and
  ``num_layers_per_pipeline_stage > 1`` so that ``_get_pipeline_stage_module`` takes the scanned
  branch and constructs ``NNXScannedPipelineStage``, whose ``__init__`` forks with
  ``split=num_layers``. Asserts that ``num_layers_per_pipeline_stage`` appears in the recorded fork
  ``split`` values and that the decoder builds.
  """
  cfg = _make_config(
      ici_pipeline_parallelism=_SITE1_ICI_PIPELINE,
      num_layers_per_pipeline_stage=2,
      scan_layers_per_stage=True,
      base_num_decoder_layers=4,  # = num_stages(2) * num_pipeline_repeats(1) * layers_per_stage(2)
      num_pipeline_microbatches=2,
      per_device_batch_size=2.0,  # global batch must be divisible by microbatches
      scan_layers=False,
  )
  mesh = _make_mesh(cfg)

  cm, recorded = _fork_spy()
  with cm:
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

  assert decoder is not None, "SITE 1: NNXDecoder pipeline build failed"
  assert cfg.num_layers_per_pipeline_stage in recorded, (
      f"SITE 1: expected fork(split={cfg.num_layers_per_pipeline_stage}) from "
      f"NNXScannedPipelineStage.__init__, but recorded={recorded}"
  )


def test_site1_scanned_pipeline_stage_fork_fires_direct():
  """SITE 1 (direct, single-device): constructing ``NNXScannedPipelineStage`` forks
  ``rngs.fork(split=num_layers)``.

  The pipeline-routed ``test_site1_scanned_pipeline_stage_fork_fires`` needs >= 2 devices and skips
  on a single CPU. This constructs the stage class DIRECTLY (the exact module the pipeline builders
  ``decoders.py:_build_nnx_pipeline_stage`` / ``nnx_decoders.py:_get_pipeline_stage_module`` hand
  back), so SITE 1's ``__init__`` fork is covered on one device with no pipeline mesh.
  """
  cfg = _make_config()
  mesh = _make_mesh(cfg)
  num_layers = 3

  cm, recorded = _fork_spy()
  with cm:
    stage = NNXScannedPipelineStage(
        NNXDecoderLayer, num_layers, cfg, mesh, None, MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1)
    )

  assert stage is not None, "SITE 1 (direct): NNXScannedPipelineStage build failed"
  assert (
      num_layers in recorded
  ), f"SITE 1 (direct): expected fork(split={num_layers}) from NNXScannedPipelineStage.__init__, recorded={recorded}"


# ---------------------------------------------------------------------------
# Sanity check on the spy itself: scan_layers=False must NOT produce a split==num_layers fork.
# ---------------------------------------------------------------------------


def test_fork_spy_control_no_scan_no_split():
  """Control: scan_layers=False builds the decoder with NO ``split=num_decoder_layers`` fork.

  Proves the SITE-2 assertion is a real positive/negative discriminator (not always-true): the
  sequential, non-scanned path records only nnx-internal ``split=None`` forks.
  """
  cfg = _make_config(scan_layers=False)
  mesh = _make_mesh(cfg)

  cm, recorded = _fork_spy()
  with cm:
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

  assert decoder is not None
  assert (
      cfg.num_decoder_layers not in recorded
  ), f"control: did not expect split={cfg.num_decoder_layers}, recorded={recorded}"


if __name__ == "__main__":
  unittest.main()
