from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError

import json, glob, os
from typing import Tuple


def _resolve_local_dir(model_path: str) -> str:
    """Return a local directory containing the model files.
    If `model_path` is a HF repo id, snapshot-download it first.
    """
    # Already a local folder with config.json?
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        return model_path

    # Otherwise treat as Hub repo id and download a snapshot
    token = os.getenv("HF_TOKEN")  # optional, but required for gated/private repos
    try:
        local_dir = snapshot_download(
            repo_id=model_path,
            token=token,
            local_dir=None,                # use HF cache
            local_dir_use_symlinks=False,  # safer on WSL/Windows drives
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "tokenizer*",
                "*.json",
                "*.txt",
                "*.model",
            ],
        )
        return local_dir
    except GatedRepoError:
        raise SystemExit(
            f"\n[ACCESS ERROR] '{model_path}' is gated.\n"
            f"Open https://huggingface.co/{model_path} in your browser and click "
            "'Access repository / Agree and access' with the same account as your token.\n"
        )
    except RepositoryNotFoundError:
        raise SystemExit(
            f"\n[NOT FOUND] '{model_path}' is not a valid repo id and no local folder with config.json was found.\n"
            "Did you mean 'google/paligemma-3b-pt-224'? Or download the model locally first.\n"
        )
    except HfHubHTTPError as e:
        raise SystemExit(f"\n[HF HUB ERROR] Could not download '{model_path}': {e}\n")


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Resolve to a local directory (download if a repo id)
    local_dir = _resolve_local_dir(model_path)

    # Load tokenizer from the resolved local dir
    token = os.getenv("HF_TOKEN")  # not strictly needed once local, but harmless
    tok_kwargs = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(local_dir, padding_side="right", **tok_kwargs)
    assert tokenizer.padding_side == "right"

    # Gather tensors from all *.safetensors shards
    tensors = {}
    for safetensors_file in glob.glob(os.path.join(local_dir, "*.safetensors")):
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load config.json from the resolved folder and build config
    with open(os.path.join(local_dir, "config.json"), "r") as f:
        config = PaliGemmaConfig(**json.load(f))

    # Instantiate model and load weights
    model = PaliGemmaForConditionalGeneration(config).to(device)
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()

    return model, tokenizer
