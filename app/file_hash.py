import hashlib


def get_file_hash(file_path):
    """
    Returns the SHA256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()