"""Read pinned Git objects and fetch only the required LFS payloads."""

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import time
from urllib.parse import quote
from urllib.request import urlopen

from .io import sha256
from .schema import SOURCE_COMMIT


class SourceStore:
    def __init__(self, repository, cache):
        self.repository = Path(repository).resolve()
        self.cache = Path(cache).resolve() / SOURCE_COMMIT
        self.records = {}
        subprocess.run(["git", "cat-file", "-e", SOURCE_COMMIT + "^{commit}"],
                       cwd=self.repository, check=True, capture_output=True)

    def fetch(self, relative):
        relative = str(relative)
        logical = PurePosixPath(relative)
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError("Source paths must stay inside the pinned repository")
        payload = subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + relative], cwd=self.repository)
        is_lfs = payload.startswith(b"version https://git-lfs.github.com/spec/v1\n")
        if is_lfs:
            pointer = dict(line.split(" ", 1) for line in payload.decode().splitlines())
            digest, size = pointer["oid"].removeprefix("sha256:"), int(pointer["size"])
        else:
            digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
        target = self.cache.joinpath(*logical.parts)
        if not (target.is_file() and target.stat().st_size == size and sha256(target) == digest):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            if is_lfs:
                url = "https://media.githubusercontent.com/media/wplso/xArm-model-training/" + SOURCE_COMMIT + "/" + quote(relative, safe="/")
                for attempt in range(3):
                    try:
                        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                            received = 0
                            while chunk := response.read(1024 * 1024):
                                received += len(chunk)
                                if received > size:
                                    raise ValueError("LFS download exceeds declared size")
                                stream.write(chunk)
                        break
                    except (OSError, TimeoutError):
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            else:
                temporary.write_bytes(payload)
            if temporary.stat().st_size != size or sha256(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise ValueError("Source integrity mismatch: " + relative)
            temporary.replace(target)
        self.records[relative] = {"sha256": digest, "bytes": size, "lfs": is_lfs}
        return target
