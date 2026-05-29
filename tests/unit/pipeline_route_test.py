# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for pipeline routing and lightweight pipeline helper behavior."""

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from flax import nnx
import jax
from jax.sharding import Mesh
import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"


def _assert_under_src(module):
  module_path = Path(module.__file__).resolve()
  assert module_path.is_relative_to(_SRC), f"{module.__name__} loaded from {module_path}"


@contextmanager
def _workspace_maxtext_modules():
  import maxtext
  import maxtext.common
  import maxtext.layers

  package_paths = (
      (maxtext, "__path__", list(maxtext.__path__)),
      (maxtext.common, "__path__", list(maxtext.common.__path__)),
      (maxtext.layers, "__path__", list(maxtext.layers.__path__)),
  )
  module_names = (
      "maxtext.common.common_types",
      "maxtext.layers.initializers",
      "maxtext.layers.nnx_wrappers",
      "maxtext.layers.pipeline",
  )
  old_sys_path = list(sys.path)
  old_maxtext_modules = {
      name: module for name, module in sys.modules.items() if name == "maxtext" or name.startswith("maxtext.")
  }
  old_attrs = {
      (maxtext.common, "common_types"): getattr(maxtext.common, "common_types", None),
      (maxtext.layers, "initializers"): getattr(maxtext.layers, "initializers", None),
      (maxtext.layers, "nnx_wrappers"): getattr(maxtext.layers, "nnx_wrappers", None),
      (maxtext.layers, "pipeline"): getattr(maxtext.layers, "pipeline", None),
  }
  missing_attrs = {
      (package, name) for package, name in old_attrs if not hasattr(package, name)
  }

  try:
    sys.path.insert(0, str(_SRC))
    maxtext.__path__ = [str(_SRC / "maxtext"), *list(maxtext.__path__)]
    maxtext.common.__path__ = [str(_SRC / "maxtext" / "common"), *list(maxtext.common.__path__)]
    maxtext.layers.__path__ = [str(_SRC / "maxtext" / "layers"), *list(maxtext.layers.__path__)]
    for name in module_names:
      sys.modules.pop(name, None)
    for package, name in old_attrs:
      if hasattr(package, name):
        delattr(package, name)

    common_types = importlib.import_module("maxtext.common.common_types")
    initializers = importlib.import_module("maxtext.layers.initializers")
    nnx_wrappers = importlib.import_module("maxtext.layers.nnx_wrappers")
    pipeline = importlib.import_module("maxtext.layers.pipeline")

    for module in (common_types, initializers, nnx_wrappers, pipeline):
      _assert_under_src(module)

    yield SimpleNamespace(
        common_types=common_types,
        initializers=initializers,
        nnx_wrappers=nnx_wrappers,
        pipeline=pipeline,
    )
  finally:
    sys.path[:] = old_sys_path
    for name in list(sys.modules):
      if name == "maxtext" or name.startswith("maxtext."):
        if name not in old_maxtext_modules:
          sys.modules.pop(name, None)
    sys.modules.update(old_maxtext_modules)
    for package, attr, path in package_paths:
      setattr(package, attr, path)
    for package_attr, value in old_attrs.items():
      package, attr = package_attr
      if package_attr in missing_attrs:
        if hasattr(package, attr):
          delattr(package, attr)
      else:
        setattr(package, attr, value)


def _minimal_config(*, pipeline_fsdp_ag_per_repeat, shard_mode):
  return SimpleNamespace(
      ici_pipeline_parallelism=1,
      dcn_pipeline_parallelism=1,
      pipeline_delay_activation_forwarding=False,
      micro_batch_size_to_train_on=1,
      num_pipeline_microbatches=1,
      num_pipeline_repeats=1,
      shard_mode=shard_mode,
      logical_axis_rules=[],
      debug_sharding=False,
      pipeline_fsdp_ag_per_repeat=pipeline_fsdp_ag_per_repeat,
      scan_layers=False,
      scan_pipeline_iterations=True,
      scan_pipeline_repeats=False,
      set_remat_policy_on_pipeline_iterations=False,
  )


class _TinyStage(nnx.Module):
  def __init__(self, rngs):
    self.weight = nnx.Param(jax.numpy.asarray(1.0))

  def __call__(self, x, *unused_args):
    return x + self.weight.value


class PipelineRouteTest(unittest.TestCase):

  def setUp(self):
    self.import_context = _workspace_maxtext_modules()
    modules = self.import_context.__enter__()
    self.addCleanup(self.import_context.__exit__, None, None, None)
    self.initializers = modules.initializers
    self.nnx_wrappers = modules.nnx_wrappers
    self.pipeline = modules.pipeline
    self.ShardMode = modules.common_types.ShardMode
    devices = jax.devices()
    self.mesh = Mesh(np.asarray(devices[:1]), ("stage",))

  def test_create_nnx_pipeline_selects_nnx_pipeline(self):
    cfg = _minimal_config(pipeline_fsdp_ag_per_repeat=False, shard_mode=self.ShardMode.AUTO)
    model = self.pipeline.create_nnx_pipeline(cfg, _TinyStage, self.mesh, rngs=nnx.Rngs(params=0))
    self.assertIsInstance(model, self.pipeline.NNXPipeline)

  def test_create_nnx_pipeline_selects_nnx_circular_pipeline(self):
    cfg = _minimal_config(pipeline_fsdp_ag_per_repeat=True, shard_mode=self.ShardMode.AUTO)
    model = self.pipeline.create_nnx_pipeline(cfg, _TinyStage, self.mesh, rngs=nnx.Rngs(params=0))
    self.assertIsInstance(model, self.pipeline.NNXCircularPipeline)

  def test_create_pipeline_selects_tolinen_pipeline(self):
    cfg = _minimal_config(pipeline_fsdp_ag_per_repeat=False, shard_mode=self.ShardMode.AUTO)
    model = self.pipeline.create_pipeline(cfg, layers=_TinyStage, mesh=self.mesh)
    self.assertIsInstance(model, self.nnx_wrappers.ToLinen)
    self.assertIs(model.module_class, self.pipeline.NNXPipeline)

  def test_create_pipeline_selects_tolinen_circular_pipeline(self):
    cfg = _minimal_config(pipeline_fsdp_ag_per_repeat=True, shard_mode=self.ShardMode.AUTO)
    model = self.pipeline.create_pipeline(cfg, layers=_TinyStage, mesh=self.mesh)
    self.assertIsInstance(model, self.nnx_wrappers.ToLinen)
    self.assertIs(model.module_class, self.pipeline.NNXCircularPipeline)

  def test_assert_only_rng_mutables_allows_rng_state(self):
    state = {"rng": nnx.RngKey(jax.random.key(0))}
    self.pipeline._assert_only_rng_mutables_in_scan_carry(state)

  def test_assert_only_rng_mutables_rejects_param(self):
    state = {"param": nnx.Param(jax.numpy.asarray(1.0))}
    with self.assertRaisesRegex(AssertionError, "Non-RngState variable"):
      self.pipeline._assert_only_rng_mutables_in_scan_carry(state)

  def test_select_circular_pipeline_core_defaults_to_nnx_scan(self):
    old = os.environ.get("MAXTEXT_CIRCULAR_PIPELINE_CORE")
    os.environ.pop("MAXTEXT_CIRCULAR_PIPELINE_CORE", None)
    try:
      self.assertEqual(self.pipeline._select_circular_pipeline_core(), "nnx_scan")
    finally:
      if old is not None:
        os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = old

  def test_select_circular_pipeline_core_rejects_unknown_value(self):
    old = os.environ.get("MAXTEXT_CIRCULAR_PIPELINE_CORE")
    os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = "bad"
    try:
      with self.assertRaisesRegex(ValueError, "Invalid MAXTEXT_CIRCULAR_PIPELINE_CORE"):
        self.pipeline._select_circular_pipeline_core()
    finally:
      if old is None:
        os.environ.pop("MAXTEXT_CIRCULAR_PIPELINE_CORE", None)
      else:
        os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = old

  def test_circular_pipeline_dispatcher_preserves_public_argument_order(self):
    old = os.environ.get("MAXTEXT_CIRCULAR_PIPELINE_CORE")
    os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = "nnx_scan"
    calls = []

    class DummyCircularPipeline:
      def _call_nnx_scan_core(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

      def _call_jax_state_core(self, *unused_args, **unused_kwargs):
        raise AssertionError("unexpected jax_state dispatch")

    try:
      result = self.pipeline.NNXCircularPipeline.__call__(
          DummyCircularPipeline(),
          "inputs",
          "segment_ids",
          "positions",
          False,
          "model_mode",
          logical_partition_spec="logical_spec",
      )
    finally:
      if old is None:
        os.environ.pop("MAXTEXT_CIRCULAR_PIPELINE_CORE", None)
      else:
        os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = old

    self.assertEqual(result, "ok")
    self.assertEqual(
        calls,
        [
            (
                ("inputs", "segment_ids", "positions", False, "model_mode"),
                {"logical_partition_spec": "logical_spec"},
            )
        ],
    )

  def test_circular_pipeline_dispatcher_routes_jax_state_core(self):
    old = os.environ.get("MAXTEXT_CIRCULAR_PIPELINE_CORE")
    os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = "jax_state"
    calls = []

    class DummyCircularPipeline:
      def _call_nnx_scan_core(self, *unused_args, **unused_kwargs):
        raise AssertionError("unexpected nnx_scan dispatch")

      def _call_jax_state_core(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    try:
      result = self.pipeline.NNXCircularPipeline.__call__(
          DummyCircularPipeline(),
          "inputs",
          "segment_ids",
          "positions",
          False,
          "model_mode",
          logical_partition_spec="logical_spec",
      )
    finally:
      if old is None:
        os.environ.pop("MAXTEXT_CIRCULAR_PIPELINE_CORE", None)
      else:
        os.environ["MAXTEXT_CIRCULAR_PIPELINE_CORE"] = old

    self.assertEqual(result, "ok")
    self.assertEqual(
        calls,
        [
            (
                ("inputs", "segment_ids", "positions", False, "model_mode"),
                {"logical_partition_spec": "logical_spec"},
            )
        ],
    )


if __name__ == "__main__":
  unittest.main()
