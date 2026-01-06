<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Convert from a Hugging Face checkpoint

MaxText provides `src/MaxText/utils/ckpt_conversion/to_maxtext.py` script, that can be used to convert a Hugging Face checkpoint to MaxText format. 
This is useful if you have a pre-trained model from Hugging Face that you want to use with MaxText for post-training.

First, make sure python3 virtual environment for MaxText is set up and enabled.
```bash
export VENV_NAME=<your virtual env name> # e.g., maxtext_venv
pip install uv
uv venv --python 3.12 --seed $VENV_NAME
source $VENV_NAME/bin/activate
```

Second, ensure you have the necessary dependencies installed (PyTorch for the conversion script).

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Setup following environment variables for conversion script

```bash
# -- Model configuration --
export HF_MODEL=<Hugging Face Model to be converted to MaxText> # e.g. 'llama3.1-8b-Instruct'
export HF_TOKEN=<Hugging Face access token> # your token to access gated HF repos

# -- MaxText configuration --
export MODEL_CHECKPOINT_DIRECTORY=<output directory to store output of checking point> # e.g., gs://my-bucket/my-checkpoint-directory

# -- storage and format options
USE_ZARR3=<Flag to use zarr3> # True to run SFT with McJAX, False to run SFT with Pathways
USE_OCDBT=<Flag to use ocdbt> # True to run SFT with McJAX, False to run SFT with Pathways
```

You can run the conversion script on a CPU machine with `hardware=cpu`

```bash
python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_CHECKPOINT_DIRECTORY} \
    scan_layers=True \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=${USE_ZARR3} \
    checkpoint_storage_use_ocdbt=${USE_OCDBT} \
    --lazy_load_tensors=true
```

For large models, it is recommended to use the `--lazy_load_tensors` flag to reduce memory usage during conversion. For example, converting a Llama3.1-70B model with `--lazy_load_tensors=true` uses around 200GB of RAM and completes in ~10 minutes.

This command will download the Hugging Face model to local machine and convert it to the MaxText format, saving it to 
`${MODEL_CHECKPOINT_DIRECTORY}/0/items` (e.g. gs://my-bucket/my-checkpoint-directory/0/items)

Set this environment variable `MAXTEXT_CKPT_PATH` to use as parameter `load_parameters_path` in the following post training sessions:

```bash
export MAXTEXT_CKPT_PATH=${MODEL_CHECKPOINT_DIRECTORY}/0/items
```

The conversion script only supports official versions of models from Hugging Face.
To see the specific models and versions currently supported for conversion, please refer to the `HF_IDS` dictionary in the MaxText utility file [here](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/utils.py).
For more info about checkpoint conversion script, please refer to [Checkpoint conversion utilities](../../guides/checkpointing_solutions/convert_checkpoint).

