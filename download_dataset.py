from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="limjiayi/hateful_memes_expanded",
    repo_type="dataset",
    local_dir="data",
    local_dir_use_symlinks=False
)

print("Dataset downloaded into ./data")