# Start from a core stack version
FROM pytorch/pytorch

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Set the timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Fetch some necessary apt packages and clean up
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --upgrade pip
RUN pip install openmim
RUN mim install mmengine mmcv

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
RUN git checkout main
RUN pip install cython --no-cache-dir
RUN FORCE_CUDA="0" 
RUN pip install --no-cache-dir -e .

# Set the default command to python
CMD ["python"]
