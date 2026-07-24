import hashlib


def get_file_hash(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while chunk := f.read(8192):

            sha.update(chunk)
    print("get file hash")
    return sha.hexdigest()