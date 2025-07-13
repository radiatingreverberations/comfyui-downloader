import os
import json
import shutil
from huggingface_hub import hf_hub_download
import requests
from .hf_utils import parse_hf_url

def download_models_from_json(json_file, output_dir="models", dry_run=False):
    os.makedirs(output_dir, exist_ok=True)

    # place HF cache inside the output directory
    hf_cache_dir = os.path.join(output_dir, "hf-cache")
    os.makedirs(hf_cache_dir, exist_ok=True)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for node in data.get("nodes", []):
        models = node.get("properties", {}).get("models", [])
        for m in models:
            url       = m.get("url")
            directory = m.get("directory", "")
            name      = m.get("name")

            target_dir = os.path.join(output_dir, directory)
            os.makedirs(target_dir, exist_ok=True)
            dest_path = os.path.join(target_dir, name)

            # Skip if destination exists (and if symlink, it must point to a real file)
            if os.path.lexists(dest_path):
                if os.path.islink(dest_path):
                    if os.path.exists(dest_path):
                        print(f"⚠️  [skip] Already exists (valid symlink): {dest_path}")
                        continue
                    else:
                        # broken symlink → remove and re-download
                        os.remove(dest_path)
                else:
                    print(f"⚠️  [skip] Already exists: {dest_path}")
                    continue

            # Dry-run: only report what would happen
            if dry_run:
                try:
                    parse_hf_url(url)
                    print(f"✔️  [dry-run] HF download: {name} → {dest_path}")
                except Exception:
                    print(f"✔️  [dry-run] HTTP download: {name} from {url} → {dest_path}")
                continue

            # Try HF API download
            try:
                repo_id, revision, file_path = parse_hf_url(url)
                cached = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    revision=revision,
                    cache_dir=hf_cache_dir
                )
                # try hard-link, then symlink, then copy
                try:
                    os.link(cached, dest_path)
                    print(f"🔗  HF hardlink: {name} → {dest_path}")
                except OSError:
                    try:
                        os.symlink(cached, dest_path)
                        print(f"🔗  HF symlink: {name} → {dest_path}")
                    except OSError:
                        shutil.copy(cached, dest_path)
                        print(f"✔️  HF download (copied): {name} → {dest_path}")

            # Fallback to plain HTTP
            except Exception:
                print(f"⏳  HTTP download: {name} from {url}")
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(dest_path, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=8192):
                        out.write(chunk)
                print(f"✔️  HTTP download done: {dest_path}")
