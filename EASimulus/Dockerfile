FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /M3

RUN apt-get update && apt-get -y install gcc

RUN apt-get update && apt-get install -yq \
        bison \
        build-essential \
        cmake \
        curl \
        flex \
        git \
        libbz2-dev \
        ninja-build \
        wget \
        tmux

RUN apt-get install ffmpeg libsm6 libxext6 -y --fix-missing

RUN python -m pip install --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./get_lpips.py .
RUN python get_lpips.py

# RUN pip install git+https://github.com/chernyadev/bigym

RUN apt-get install -y libgl1-mesa-glx libosmesa6

RUN pip install craftax
RUN pip install numpy==1.25.2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install -y libgl1-mesa-dev libosmesa6-dev
# RUN pip install git+https://github.com/leor-c/Kinetix-CPU.git

#RUN groupadd -r rem_users && useradd -r -g rem_users rem_user
#USER rem_user
