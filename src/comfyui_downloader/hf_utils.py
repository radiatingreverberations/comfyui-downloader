from urllib.parse import urlparse


def parse_hf_url(url):
    """
    Parse Hugging Face URLs of the form:
    https://huggingface.co/{repo_id}/resolve/{revision}/{path}
    Returns (repo_id, revision, file_path)
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        raise ValueError("Not a HF resolve URL")
    if parsed.netloc != "huggingface.co":
        raise ValueError("Not a valid Hugging Face URL")
    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3]
    file_path = "/".join(parts[4:])
    return repo_id, revision, file_path
