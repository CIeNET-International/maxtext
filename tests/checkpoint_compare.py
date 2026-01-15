import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state
import numpy as np
from typing import Any, Dict, Tuple, List
from jax.tree_util import PyTreeDef

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
        key_str = jax.tree_util.keystr(path)
        if 'token_embedder' not in key_str and leaf.ndim >= 2:
            print(f"TRANSPOSING: {key_str} with shape {leaf.shape}")
            # New axes: (1, 0, 2, 3, ..., ndim-1)
            axes = (1, 0) + tuple(range(2, leaf.ndim))
            return jnp.transpose(leaf, axes=axes)
        else:
            if 'token_embedder' in key_str:
                print(f"SKIPPING: {key_str} because it is token_embedder")
            else:
                print(f"SKIPPING: {key_str} with shape {leaf.shape} (ndim < 2)")
            return leaf
    print("Applying transformations to NNX params...")
    return jax.tree_util.tree_map_with_path(_transform, nnx_params)

def compare_params(params1: Dict[str, Any], params2: Dict[str, Any], prefix: str = "") -> bool:
    """Compares two PyTrees of parameters."""
    struct1 = jax.tree_util.tree_structure(params1)
    struct2 = jax.tree_util.tree_structure(params2)
    if struct1 != struct2:
        print(f"[{prefix}] Tree structures differ.")
        # For more detailed structure diff, you might need a different tool
        return False
    print(f"[{prefix}] Tree structures are the same.")

    all_match = True

    def compare_leaf_with_path(path: Tuple[jax.tree_util.DictKey, ...], x: jax.Array, y: jax.Array):
        nonlocal all_match
        key_str = jax.tree_util.keystr(path)

        if x.shape != y.shape:
            print(f"[{prefix}{key_str}] Shapes differ: {x.shape} vs {y.shape}")
            all_match = False
        elif x.dtype != y.dtype:
            print(f"[{prefix}{key_str}] Dtypes differ: {x.dtype} vs {y.dtype}")
            all_match = False
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

    jax.tree_util.tree_map_with_path(compare_leaf_with_path, params1, params2)
    return all_match

linen_ckpt_path = "gs://maxtext-test/gemma2-hf-to-mt/2b/scanned/0/items/0/items"
nnx_ckpt_path = "gs://maxtext-test/gemma2-hf-to-mt-NNX-decoder/2b/scanned/0/items"
print("Loading Linen params...")
linen_params = load_checkpoint_params(linen_ckpt_path)
print("Loading NNX params...")
nnx_params = load_checkpoint_params(nnx_ckpt_path)

if linen_params is not None and nnx_params is not None:
    # Transform NNX params to match expected Linen shapes
    nnx_params_transformed = transform_nnx_params(nnx_params)

    print("\nComparing Linen params with Transformed NNX params...")
    if compare_params(linen_params, nnx_params_transformed):
        print("\nCheckpoints are the same after transformation!")
    else:
        print("\nCheckpoints DIFFER after transformation.")
else:
    print("Failed to load params from one or both checkpoints.")


