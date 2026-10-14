import hashlib

import pytest

from scripts.download_data import ChecksumError, sha256_of, verify_checksum


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert sha256_of(p) == hashlib.sha256(b"hello world").hexdigest()


def test_verify_checksum_accepts_and_rejects(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    verify_checksum(p, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(ChecksumError):
        verify_checksum(p, "0" * 64)
