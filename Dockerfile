# LiveTalking image, built from the cloned source only:
# official NVIDIA CUDA base, standard PyPI and PyTorch indexes, no prebuilt images,
# no Aliyun mirrors. Model weights and avatars are NOT baked in; they are mounted.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential pkg-config \
        ffmpeg libglib2.0-0 libgl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# PyTorch 2.9.1 / CUDA 12.8, as the LiveTalking README specifies.
# torch >= 2.6 defaults torch.load(weights_only=True), which refuses code in .pth files.
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN sed -i 's/\r$//' /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt picklescan

# Cap x264 frame size (VBV) so mid-stream keyframes don't burst to ~350 KB and lose packets on Wi-Fi.
RUN sed -i 's/"tune": "zerolatency",\?/"tune": "zerolatency", "x264-params": "vbv-maxrate=3000:vbv-bufsize=500",/' /opt/venv/lib/python3.12/site-packages/aiortc/codecs/h264.py \
 && grep -q 'vbv-maxrate' /opt/venv/lib/python3.12/site-packages/aiortc/codecs/h264.py

COPY . /app
# Windows checkout has CRLF line endings; normalise text files that Linux tools read.
RUN find /app -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.json' \) \
        -exec sed -i 's/\r$//' {} +

# S3FD face detector (only used when generating avatars), from its original author.
# The torch-hub file name embeds the first 10 hex chars of its SHA-256; the build fails if it doesn't match.
RUN curl -fsSL -o /app/avatars/wav2lip/face_detection/detection/sfd/s3fd.pth \
        https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
    && sha256sum /app/avatars/wav2lip/face_detection/detection/sfd/s3fd.pth | grep -q '^619a316812'

# Run as the TrueNAS "apps" user, not root.
RUN mkdir -p /app/models /app/data/avatars /app/data/record \
    && chown -R 568:568 /app
USER 568:568

EXPOSE 8010
# --stun "" : LAN only, no public STUN server. --max_session 2 : avatar + a test browser.
ENTRYPOINT ["python", "app.py", "--transport", "webrtc", "--model", "wav2lip", "--stun", "", "--max_session", "2"]
CMD ["--avatar_id", "sylvia"]
