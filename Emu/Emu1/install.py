import os
os.environ["HF_HOME"] = "../../.cache"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/Emu",
    repo_type="model",
    local_dir="../../.cache/BAAI/Emu"
)
