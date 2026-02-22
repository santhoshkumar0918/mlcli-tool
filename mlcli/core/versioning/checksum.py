"""Checksum utilities for artifact integrity verification."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5, sha1)
        
    Returns:
        Hexadecimal checksum string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_dict_checksum(data: Dict[str, Any], algorithm: str = "sha256") -> str:
    """Compute checksum of a JSON-serializable dictionary.
    
    Args:
        data: Dictionary to hash
        algorithm: Hash algorithm (sha256, md5, sha1)
        
    Returns:
        Hexadecimal checksum string
    """
    hash_func = hashlib.new(algorithm)
    
    serialized = json.dumps(data, sort_keys=True, default=str)
    hash_func.update(serialized.encode("utf-8"))
    
    return hash_func.hexdigest()


def compute_string_checksum(text: str, algorithm: str = "sha256") -> str:
    """Compute checksum of a string.
    
    Args:
        text: String to hash
        algorithm: Hash algorithm (sha256, md5, sha1)
        
    Returns:
        Hexadecimal checksum string
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(text.encode("utf-8"))
    
    return hash_func.hexdigest()


def verify_file_checksum(file_path: Path, expected_checksum: str, algorithm: str = "sha256") -> bool:
    """Verify a file's checksum matches expected value.
    
    Args:
        file_path: Path to the file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm used
        
    Returns:
        True if checksums match, False otherwise
    """
    try:
        actual = compute_file_checksum(file_path, algorithm)
        return actual == expected_checksum
    except Exception:
        return False


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0
    return file_path.stat().st_size


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
