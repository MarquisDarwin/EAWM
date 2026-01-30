import hashlib
import os

import requests
from pathlib import Path
from torchvision import models
from loguru import logger
from tqdm import tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}


CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        logger.info(f"LPIPS model not found in '{path}'.")
        logger.info("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    else:
        logger.info(f"Found {name} model weights!")
    return path


def get_lpips():
    # get vgg16 weights:
    models.vgg16(pretrained=True)

    # get the additional weights from IRIS:
    project_root = Path.cwd()

    ckpt_path = project_root / "cache" / "rem" / "tokenizer_pretrained_vgg"
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    get_ckpt_path('vgg_lpips', ckpt_path)


if __name__ == '__main__':
    get_lpips()



