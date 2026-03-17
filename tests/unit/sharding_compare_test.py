# Copyright 2023–2026 Google LLC
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

"""Compare expected sharding of models with actual sharding of models."""

import itertools
import hashlib
import json
import os
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import clear_input_shardings_dump
# import optax

from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train.train_compile import get_shaped_inputs, get_topology_mesh, validate_config
from tests.utils.sharding_dump import load_json, input_sharding_to_json, named_shardings_to_json, partition_specs_to_json
from tests.utils.sharding_dump import save_json
from tests.utils.test_helpers import get_test_config_path
import pytest

Transformer = models.transformer_as_linen



MODEL_NAMES = [
    # "default",
    # "llama2-7b",
    # "llama2-13b",
    # "llama2-70b",
    # "llama3-8b",
    # "llama3-70b",
    # "llama3.1-8b",
    # "llama3.1-70b",
    # "llama3.1-405b",
    # "llama3.3-70b",
    # "mistral-7b",
    # "mixtral-8x7b",
    # "mixtral-8x22b",
    "deepseek2-16b",
    # "deepseek2-236b",
    # "deepseek3-671b",
    # "deepseek3-671b-2dfsdp",
    # "deepseek3-test",
    # "deepseek3-tiny",
    # "deepseek3.2-671b",
    # "gemma-7b",
    # "gemma-2b",
    # "gemma2-2b",
    # "gemma2-9b",
    # "gemma2-27b",
    # "gemma3-4b",
    # "gemma3-12b",
    # "gemma3-27b",
    "qwen3-0.6b",
    # "qwen3-4b",
    # "qwen3-4b-thinking-2507",
    # "qwen3-8b",
    # "qwen3-14b",
    # "qwen3-32b",
    # "qwen3-235b-a22b",
    # "qwen3-30b-a3b",
    # "qwen3-480b-a35b",
    # "qwen3-next-80b-a3b",
    # "qwen3-omni-30b-a3b",
    # "gpt3-175b",
    # "gpt3-22b",
    # "gpt3-6b",
    # "gpt3-52k",
    "gpt-oss-20b",
    # "gpt-oss-120b",
    # "llama4-17b-16e",
    # "llama4-17b-128e",
]

TOPOLOGIES = [
    # "tpu7x-2",
    # "tpu7x-8",
    #"tpu7x-16",
    # "tpu7x-32",
    # "tpu7x-64",
    # "tpu7x-128",
    # "tpu7x-256",
    # "tpu7x-384",
    # "tpu7x-512",
    # "tpu7x-640",
    # "tpu7x-768",
    # "tpu7x-896",
    # "tpu7x-1024",
    # "tpu7x-1152",
    # "tpu7x-1280",
    # "tpu7x-1408",
    # "tpu7x-1536",
    # "tpu7x-1664",
    # "tpu7x-1792",
    # "tpu7x-1920",
    # "tpu7x-2048",
    # "tpu7x-2176",
    # "tpu7x-2304",
    # "tpu7x-2432",
    # "tpu7x-2560",
    # "tpu7x-2688",
    # "tpu7x-2816",
    # "tpu7x-2944",
    # "tpu7x-3072",
    # "tpu7x-3200",
    # "tpu7x-3328",
    # "tpu7x-3456",
    # "tpu7x-3584",
    # "tpu7x-3712",
    # "tpu7x-3840",
    # "tpu7x-3968",
    # "tpu7x-4096",
    # "tpu7x-4224",
    # "tpu7x-4352",
    # "tpu7x-4480",
    # "tpu7x-4608",
    # "tpu7x-4736",
    # "tpu7x-4864",
    # "tpu7x-4992",
    # "tpu7x-5120",
    # "tpu7x-5248",
    # "tpu7x-5376",
    # "tpu7x-5504",
    # "tpu7x-5632",
    # "tpu7x-5760",
    # "tpu7x-5888",
    # "tpu7x-6016",
    # "tpu7x-6144",
    # "tpu7x-6272",
    # "tpu7x-6400",
    # "tpu7x-6528",
    # "tpu7x-6656",
    # "tpu7x-6784",
    # "tpu7x-6912",
    # "tpu7x-7040",
    # "tpu7x-7168",
    # "tpu7x-7296",
    # "tpu7x-7424",
    # "tpu7x-7552",
    # "tpu7x-7680",
    # "tpu7x-7808",
    # "tpu7x-7936",
    # "tpu7x-8064",
    # "tpu7x-8192",
    # "tpu7x-8320",
    # "tpu7x-8448",
    # "tpu7x-8704",
    # "tpu7x-8832",
    # "tpu7x-8960",
    # "tpu7x-9216",
    # "tpu7x-9472",
    # "tpu7x-9600",
    # "tpu7x-9728",
    # "tpu7x-9856",
    # "tpu7x-9984",
    # "tpu7x-10240",
    # "tpu7x-10368",
    # "tpu7x-10496",
    # "tpu7x-10752",
    # "tpu7x-10880",
    # "tpu7x-11008",
    # "tpu7x-11136",
    # "tpu7x-11264",
    # "tpu7x-11520",
    # "tpu7x-11648",
    # "tpu7x-11776",
    # "tpu7x-11904",
    # "tpu7x-12032",
    # "tpu7x-12160",
    # "tpu7x-12288",
    # "tpu7x-13824",
    # "tpu7x-16384",
    # "tpu7x-17920",
    # "tpu7x-18432",
    # "v6e-1",
    # "v6e-4",
    # "v6e-8",
    "v6e-16",
    # "v6e-32",
    # "v6e-64",
    # "v6e-128",
    # "v6e-256",
    # "v5e-1",
    # "v5e-4",
    # "v5e-8",
    # "v5e-16",
    # "v5e-32",
    # "v5e-64",
    # "v5e-128",
    # "v5e-256",
    # "v4-8",
    # "v4-16",
    # "v4-32",
    # "v4-64",
    # "v4-128",
    # "v4-256",
    # "v4-384",
    # "v4-512",
    # "v4-1024",
    # "v4-1536",
    # "v4-2048",
    # "v4-4096",
    # "v5p-8",
    #"v5p-16",
    # "v5p-32",
    # "v5p-64",
    # "v5p-128",
    # "v5p-256",
    # "v5p-384",
    # "v5p-512",
    # "v5p-640",
    # "v5p-768",
    # "v5p-896",
    # "v5p-1024",
    # "v5p-1152",
    # "v5p-1280",
    # "v5p-1408",
    # "v5p-1536",
    # "v5p-1664",
    # "v5p-1792",
    # "v5p-1920",
    # "v5p-2048",
    # "v5p-2176",
    # "v5p-2304",
    # "v5p-2432",
    # "v5p-2560",
    # "v5p-2688",
    # "v5p-2816",
    # "v5p-2944",
    # "v5p-3072",
    # "v5p-3200",
    # "v5p-3328",
    # "v5p-3456",
    # "v5p-3584",
    # "v5p-3712",
    # "v5p-3840",
    # "v5p-3968",
    # "v5p-4096",
    # "v5p-4224",
    # "v5p-4352",
    # "v5p-4480",
    # "v5p-4608",
    # "v5p-4736",
    # "v5p-4864",
    # "v5p-4992",
    # "v5p-5120",
    # "v5p-5248",
    # "v5p-5376",
    # "v5p-5504",
    # "v5p-5632",
    # "v5p-5760",
    # "v5p-5888",
    # "v5p-6016",
    # "v5p-6144",
    # "v5p-6272",
    # "v5p-6400",
    # "v5p-6528",
    # "v5p-6656",
    # "v5p-6784",
    # "v5p-6912",
    # "v5p-7040",
    # "v5p-7168",
    # "v5p-7296",
    # "v5p-7424",
    # "v5p-7552",
    # "v5p-7680",
    # "v5p-7808",
    # "v5p-7936",
    # "v5p-8064",
    # "v5p-8192",
    # "v5p-8320",
    # "v5p-8448",
    # "v5p-8704",
    # "v5p-8832",
    # "v5p-8960",
    # "v5p-9216",
    # "v5p-9472",
    # "v5p-9600",
    # "v5p-9728",
    # "v5p-9856",
    # "v5p-9984",
    # "v5p-10240",
    # "v5p-10368",
    # "v5p-10496",
    # "v5p-10752",
    # "v5p-10880",
    # "v5p-11008",
    # "v5p-11136",
    # "v5p-11264",
    # "v5p-11520",
    # "v5p-11648",
    # "v5p-11776",
    # "v5p-11904",
    # "v5p-12032",
    # "v5p-12160",
    # "v5p-12288",
    # "v5p-13824",
    # "v5p-17920",
    # "a3"
]

SLICES = [1, 4]

TEST_CASES = list(itertools.product(MODEL_NAMES, TOPOLOGIES, SLICES))



def compute_checksum(d: dict) -> str:
  """Compute a checksum (SHA256) of a dictionary."""
  # Serialize the dictionary into a JSON string (ensuring consistent ordering of keys)
  json_str = json.dumps(d, sort_keys=True)

  # Compute the SHA256 checksum of the serialized string
  checksum = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

  return checksum


def compare_sharding_jsons(json1: dict, model1_name: str, json2: dict, model2_name: str) -> bool:
  """Compare two json files and print the differences if any."""
  keys1 = set(json1.keys())
  keys2 = set(json2.keys())

  only_in_1 = keys1 - keys2
  only_in_2 = keys2 - keys1
  shared_keys = keys1 & keys2

  has_diff = False

  if only_in_1:
    print(f"Keys only in {model1_name}:")
    for k in sorted(only_in_1):
      print(f"  {k}")
    has_diff = True

  if only_in_2:
    print(f"Keys only in {model2_name}:")
    for k in sorted(only_in_2):
      print(f"  {k}")
    has_diff = True

  for key in sorted(shared_keys):
    entry1 = json1[key]
    entry2 = json2[key]

    if isinstance(entry1, dict) and isinstance(entry2, dict):
      mesh1 = entry1.get("mesh", {})
      mesh2 = entry2.get("mesh", {})

      spec1 = entry1.get("partition_spec", [])
      spec2 = entry2.get("partition_spec", [])

      shape1 = entry1.get("shape")
      shape2 = entry2.get("shape")

      if mesh1 != mesh2:
        print(f"\nMesh mismatch at '{key}':")
        print(f"  {model1_name}: {mesh1}")
        print(f"  {model2_name}: {mesh2}")
        has_diff = True

      if spec1 != spec2:
        print(f"\nPartitionSpec mismatch at '{key}':")
        print(f"  {model1_name}: {spec1}")
        print(f"  {model2_name}: {spec2}")
        has_diff = True

      if shape1 != shape2:
        print(f"\nShape mismatch at '{key}':")
        print(f"  {model1_name}: {shape1}")
        print(f"  {model2_name}: {shape2}")
        has_diff = True

    else:
      print(f"\nFormat mismatch at '{key}':")
      print(f"  {model1_name} type: {type(entry1)}")
      print(f"  {model2_name} type: {type(entry2)}")
      has_diff = True

  return has_diff


# Requires JAX TPU support to generate the simulated TPU topology.
@pytest.mark.cpu_only
@pytest.mark.tpu_backend
@pytest.mark.parametrize("model_name, topology, num_slice", TEST_CASES)
def test_sharding_dump_for_model(model_name: str, topology: str, num_slice: str) -> None:
  """
  Test sharding configurations from train_compile.get_shaped_inputs.
  This test verifies that the sharding configurations for various models and topologies remain consistent with golden files.
  """
  params = [
      "/deps/MaxText/tests/unit/sharding_compare_test",
      get_test_config_path(),
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
      "log_config=false",
      "debug_sharding=true",  # for input sharding dump
  ]

  root_dir = "tests/utils/sharding_info"
  base_path = os.path.join(root_dir, model_name, topology, f"slice_{num_slice}")

  named_json_path = os.path.join(base_path, "named_shardings.json")
  logical_json_path = os.path.join(base_path, "logical_shardings.json")
  input_json_path = os.path.join(base_path, "input_shardings.json")

  if not os.path.exists(named_json_path):
    pytest.skip(f"Missing named_shardings.json for {model_name} {topology} slice {num_slice}")
    return
  if not os.path.exists(logical_json_path):
    pytest.skip(f"Missing logical_shardings.json for {model_name} {topology} slice {num_slice}")
    return
  if not os.path.exists(input_json_path):
    pytest.skip(f"Missing input_shardings.json for {model_name} {topology} slice {num_slice}")
    return

  config = pyconfig.initialize(params)
  validate_config(config)

  clear_input_shardings_dump()
  topology_mesh = get_topology_mesh(config)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  optimizers.get_optimizer(config, learning_rate_schedule)
  shaped_train_args, _, state_mesh_shardings, logical_shardings, _ = get_shaped_inputs(topology_mesh, config)

  error_messages = []

  # 1. Compare Named Shardings
  actual_named = named_shardings_to_json(state_mesh_shardings, shaped_train_args[0])
  expected_named = load_json(named_json_path)
  # calculate checksum
  actual_named_sum = compute_checksum(actual_named)
  expected_named_sum = compute_checksum(expected_named)
  named_match = actual_named_sum == expected_named_sum

  if not named_match:
    print(f"\n[FAIL] Physical Sharding Mismatch: {model_name} {topology} slice {num_slice}", flush=True)
    compare_sharding_jsons(expected_named, "Expected (Physical)", actual_named, "Actual (Physical)")
    error_messages.append(f" Physical sharding mismatch for {model_name} on {topology} slice {num_slice}")

  # 2. Compare Logical Shardings
  actual_logical = partition_specs_to_json(logical_shardings, shaped_train_args[0])
  expected_logical = load_json(logical_json_path)
  # calculate checksum
  actual_logical_sum = compute_checksum(actual_logical)
  expected_logical_sum = compute_checksum(expected_logical)
  logical_match = actual_logical_sum == expected_logical_sum

  if not logical_match:
    print(f"\n[FAIL] Logical Sharding Mismatch: {model_name} {topology} slice {num_slice}", flush=True)
    compare_sharding_jsons(expected_logical, "Expected (Logical)", actual_logical, "Actual (Logical)")
    error_messages.append(f"Logical sharding mismatch for {model_name} on {topology} slice {num_slice}")

  # 3. Compare Input Shardings
  actual_input = input_sharding_to_json()
  json_path_input = os.path.join(base_path, "input_shardings_actual.json")
  save_json(json_path_input , actual_input)
  expected_input = load_json(input_json_path)
  # calculate checksum
  actual_input_sum = compute_checksum(actual_input)
  expected_input_sum = compute_checksum(expected_input)

  print(f"actual_input_sum {actual_input_sum}")
  print(f"expected_input_sum {expected_input_sum}")
  input_match = actual_input_sum == expected_input_sum

  if not input_match:
    print(f"\n[FAIL] Input Sharding Mismatch: {model_name} {topology} slice {num_slice}", flush=True)
    # compare_sharding_jsons(expected_input, "Expected (Input)", actual_input, "Actual (Input)")
    error_messages.append(f"Input sharding mismatch for {model_name} on {topology} slice {num_slice}")

  assert not error_messages, "\n".join(error_messages)


@pytest.fixture(
    scope="module",
    params=[pytest.param(case, id=f"{case[0]}-{case[1]}-{case[2]}") for case in TEST_CASES],
)
def abstract_state_and_shardings(request):
  """Pytest fixture to set up model, config, and generate abstract state once per test case."""
  model_name, topology, num_slice = request.param
  print(f"Testing model: {model_name}, topology: {topology}, num_slices: {num_slice}", flush=True)
  params = [
      "/deps/MaxText/tests/unit/sharding_compare_test",
      get_test_config_path(),
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slice}",
      f"model_name={model_name}",
      "weight_dtype=float32",
  ]
  config = pyconfig.initialize(params)
  validate_config(config)

  topology_mesh = get_topology_mesh(config)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=topology_mesh, quant=quant)

  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # tx = optax.adam(learning_rate=learning_rate_schedule)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  rng = jax.random.PRNGKey(0)

  # Get abstract state and physical shardings from maxtext_utils
  abstract_state, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
      model, tx, config, rng, topology_mesh, is_training=True
  )

  # Get logical shardings from maxtext_utils
  logical_shardings = maxtext_utils.get_logical_annotations(model, tx, config, rng, topology_mesh, is_training=True)

  return model_name, topology, num_slice, abstract_state, state_mesh_shardings, logical_shardings


@pytest.mark.cpu_only
@pytest.mark.tpu_backend
class TestGetAbstractState:
  """Test class for get_abstract_state function and sharding comparison."""

  # Requires JAX TPU support to generate the simulated TPU topology.
  def test_get_abstract_state_sharding(self, abstract_state_and_shardings):  # pylint: disable=redefined-outer-name
    """Tests that get_abstract_state returns a state with the correct abstract structure and compares sharding."""

    model_name, topology, num_slice, abstract_state, state_mesh_shardings, logical_shardings = (
        abstract_state_and_shardings
    )

    assert hasattr(abstract_state, "params")
    assert hasattr(abstract_state, "opt_state")
    param_leaf = jax.tree_util.tree_leaves(abstract_state.params)[0]
    assert isinstance(param_leaf, jax.ShapeDtypeStruct)
    assert param_leaf.dtype == jnp.float32

    root_dir = "tests/utils/sharding_info"  # Or your target directory
    base_path = os.path.join(root_dir, model_name, topology, f"slice_{num_slice}")
    os.makedirs(base_path, exist_ok=True)  # Ensure directory exists for saving actual

    error_messages = []

    # 1. Compare Physical/Named Shardings
    named_json_path = os.path.join(base_path, "named_shardings.json")
    if not os.path.exists(named_json_path):
      pytest.skip(f"Missing named_shardings.json for {model_name} {topology} slice {num_slice}")
      return

    # Use state_mesh_shardings from the fixture
    actual_named = named_shardings_to_json(state_mesh_shardings, abstract_state)
    expected_named = load_json(named_json_path)

    if compare_sharding_jsons(expected_named, "Expected (Physical)", actual_named, "Actual (Physical)"):
      error_messages.append(f"Physical sharding mismatch for {model_name} on {topology} slice {num_slice}")

    # 2. Compare Logical Shardings
    logical_json_path = os.path.join(base_path, "logical_shardings.json")
    if not os.path.exists(logical_json_path):
      pytest.skip(f"Missing logical_shardings.json for {model_name} {topology} slice {num_slice}")
      return

    # Use logical_shardings from the fixture
    actual_logical = partition_specs_to_json(logical_shardings, abstract_state)
    expected_logical = load_json(logical_json_path)

    if compare_sharding_jsons(expected_logical, "Expected (Logical)", actual_logical, "Actual (Logical)"):
      error_messages.append(f"Logical sharding mismatch for {model_name} on {topology} slice {num_slice}")

    assert not error_messages, "\n".join(error_messages)
