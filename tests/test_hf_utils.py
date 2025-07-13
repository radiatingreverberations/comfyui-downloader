import pytest
from comfyui_downloader.hf_utils import parse_hf_url

def test_parse_valid_url():
    url = "https://huggingface.co/owner/repo/resolve/main/path/to/file.txt"
    repo_id, revision, file_path = parse_hf_url(url)
    assert repo_id == "owner/repo"
    assert revision == "main"
    assert file_path == "path/to/file.txt"

@pytest.mark.parametrize("url", [
    "https://huggingface.co/owner/repo/main/path/to/file.txt",  # missing resolve
    "https://example.com/owner/repo/resolve/main/file.txt",     # wrong domain
    "not a url"
])
def test_parse_invalid_url(url):
    with pytest.raises(ValueError):
        parse_hf_url(url)

# Test edge cases

def test_parse_url_with_long_path():
    url = "https://huggingface.co/owner/repo/resolve/v1.0.0/a/b/c/d/e/f.txt"
    repo_id, revision, file_path = parse_hf_url(url)
    assert repo_id == "owner/repo"
    assert revision == "v1.0.0"
    assert file_path == "a/b/c/d/e/f.txt"

def test_parse_url_with_root_file():
    url = "https://huggingface.co/owner/repo/resolve/develop/file.bin"
    repo_id, revision, file_path = parse_hf_url(url)
    assert file_path == "file.bin"
