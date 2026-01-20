import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state
import numpy as np
from typing import Any, Dict, Tuple, List, Sequence, Optional
from jax.tree_util import PyTreeDef, tree_flatten_with_path, keystr
from absl import app
from absl import flags
import collections

# Define command-line flags for the checkpoint paths
_LINEN_CKPT_PATH = flags.DEFINE_string(
    'linen_ckpt_path', None, 'Path to the Linen model checkpoint items directory.', required=True)
_NNX_CKPT_PATH = flags.DEFINE_string(
    'nnx_ckpt_path', None, 'Path to the NNX model checkpoint items directory.', required=True)

def load_checkpoint_params(path: str) -> Dict[str, Any]:
    """Loads parameters from an Orbax checkpoint path."""
    print(f"Loading checkpoint from: {path}")
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(path)
    if isinstance(restored_state, dict) and 'params' in restored_state:
        return restored_state['params']
    return restored_state

def transform_nnx_params(nnx_params: Dict[str, Any]) -> Dict[str, Any]:
    """Applies specific transformations to the NNX parameter tree."""

    def _transform(path: Tuple[jax.tree_util.DictKey, ...], leaf: jax.Array) -> jax.Array:
        key_str = keystr(path)
        if 'layers' in key_str and leaf.ndim >= 2:
            print(f"TRANSPOSING: {key_str} with shape {leaf.shape}")
            axes = (1, 0) + tuple(range(2, leaf.ndim))
            return jnp.transpose(leaf, axes=axes)
        else:
            if 'token_embedder' in key_str:
                print(f"SKIPPING Transpose: {key_str} because it is token_embedder")
            else:
                print(f"SKIPPING Transpose: {key_str} with shape {leaf.shape} (ndim < 2)")
            return leaf
    print("Applying transformations to NNX params...")
    return jax.tree_util.tree_map_with_path(_transform, nnx_params)

def get_tree_structure_info(tree: Dict[str, Any]) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Creates a map from stringified paths to (shape, dtype) for each leaf."""
    flat_with_path, _ = tree_flatten_with_path(tree)
    structure_info = {}
    for path, leaf in flat_with_path:
        key_s = keystr(path)
        shape = getattr(leaf, 'shape', 'N/A')
        dtype = str(getattr(leaf, 'dtype', type(leaf).__name__))
        structure_info[key_s] = (shape, dtype)
    return structure_info

def print_structure_diff(info1: Dict[str, Any], info2: Dict[str, Any], name1: str, name2: str):
    """Prints differences between two tree structure info dicts."""
    keys1 = set(info1.keys())
    keys2 = set(info2.keys())

    added = keys2 - keys1
    if added:
        print(f"\nKeys added in {name2}:")
        for k in sorted(added):
            print(f"  + {k}: {info2[k]}")

    removed = keys1 - keys2
    if removed:
        print(f"\nKeys removed from {name1} (not in {name2}):")
        for k in sorted(removed):
            print(f"  - {k}: {info1[k]}")

    common = keys1.intersection(keys2)
    changed = []
    for k in sorted(common):
        if info1[k] != info2[k]:
            changed.append(k)

    if changed:
        print(f"\nKeys with different shape/dtype in {name1} vs {name2}:")
        for k in changed:
            print(f"  ~ {k}: {info1[k]} -> {info2[k]}")

def compare_params(params1: Dict[str, Any], params2: Dict[str, Any], prefix: str = "") -> bool:
    """Compares two PyTrees of parameters."""
    struct1 = jax.tree_util.tree_structure(params1)
    struct2 = jax.tree_util.tree_structure(params2)

    if struct1 != struct2:
        print(f"[{prefix}] Tree structures differ.")
        info1 = get_tree_structure_info(params1)
        info2 = get_tree_structure_info(params2)
        print_structure_diff(info1, info2, "Linen", "NNX")
        return False # Stop comparison if structures are fundamentally different
    print(f"[{prefix}] Tree structures are the same.")

    all_match = True

    def compare_leaf_with_path(path: Tuple[jax.tree_util.DictKey, ...], x: jax.Array, y: jax.Array):
        nonlocal all_match
        key_str = keystr(path)

        # Shape and dtype equality are already guaranteed by the structure check above
        # if struct1 == struct2 for jax trees.

        # Always calculate and print numerical differences for numeric arrays
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            x_np, y_np = np.asarray(x), np.asarray(y)
            abs_diff = np.abs(x_np - y_np)
            mean_diff = np.mean(abs_diff)
            max_diff = np.max(abs_diff)
            is_close = np.allclose(x_np, y_np)

            print(f"[{prefix}{key_str}] "
                  f"Mean abs diff: {mean_diff:.2e}, "
                  f"Max abs diff: {max_diff:.2e}, "
                  f"AllClose: {is_close}")

            if not is_close:
                all_match = False
        else:
            # Handle non-numerical types if any
            is_equal = np.array_equal(np.asarray(x), np.asarray(y))
            print(f"[{prefix}{key_str}] Non-numeric. Equal: {is_equal}")
            if not is_equal:
                all_match = False

    jax.tree_util.tree_map_with_path(compare_leaf_with_path, params1, params2)
    return all_match

def main(argv: Sequence[str]):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    linen_ckpt_path = _LINEN_CKPT_PATH.value
    nnx_ckpt_path = _NNX_CKPT_PATH.value

    print(f"Linen Checkpoint Path: {linen_ckpt_path}")
    print(f"NNX Checkpoint Path: {nnx_ckpt_path}")

    print("Loading Linen params...")
    linen_params = load_checkpoint_params(linen_ckpt_path)
    print("Loading NNX params...")
    nnx_params = load_checkpoint_params(nnx_ckpt_path)

    if linen_params is not None and nnx_params is not None:
        nnx_params_transformed = transform_nnx_params(nnx_params)

        print("\nComparing Linen params with Transformed NNX params...")
        if compare_params(linen_params, nnx_params_transformed):
            print("\nCheckpoints are considered the same (within np.allclose tolerance) after transformation!")
        else:
            print("\nCheckpoints DIFFER after transformation.")
    else:
        print("Failed to load params from one or both checkpoints.")

if __name__ == '__main__':
    app.run(main)

