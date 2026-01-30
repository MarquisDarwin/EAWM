import os

import requests
from filelock import FileLock
from tqdm import tqdm
def download_with_progress_bar(
    url: str, save_path: str, desc: str = "", overwrite: bool = False
):
    #if os.path.exists(save_path) and not overwrite:
    #    print(f"File already exists: {save_path}")
    #    return
    if os.path.exists(save_path):
        temp_size = os.path.getsize(save_path)
    else:
        temp_size=0
    with open(save_path, "ab") as f:
        print(f"Downloading {save_path}")
        print(url)
        headers = {'Range': 'bytes=%d-' % temp_size,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0"}
        response = requests.get(url, stream=True, headers=headers)
        total_length = response.headers.get("content-length")

        content_type = response.headers.get("content-type")
        if content_type is not None and content_type.startswith("text/html"):
            if temp_size>0:
                print("download finished")
            else:
                print(f"Invalid URL: {url}")
            return
            
        #response = requests.get(url, stream=True, headers=headers)
        if total_length is None:  # no content length header
            return
        else:
            dl = temp_size
            total_length = int(total_length)
            with tqdm(total=total_length+temp_size, unit="B", unit_scale=True, desc=desc) as pbar:
                pbar.update(dl)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    pbar.update(len(data))
download_with_progress_bar("https://pub-bebbada739114fa1aa96aaf25c873a66.r2.dev/all_type/PickupType.tar.gz","test/data")