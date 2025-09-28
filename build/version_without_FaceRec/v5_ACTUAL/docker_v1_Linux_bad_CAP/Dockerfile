FROM mambaorg/micromamba:1.5.10

WORKDIR /app

# --- system deps (root) ---
USER root
RUN install -d -m 0755 /var/lib/apt/lists/partial && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg v4l-utils \
      build-essential cmake pkg-config \
      curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

USER mambauser

# ======== Conda env ========
# <-- ВАЖНО: берем файл из unified_api/environment.yml -->
COPY unified_api/environment.yml /tmp/environment.yml

RUN grep -vE '^(name|prefix|envname):' /tmp/environment.yml > /tmp/environment.clean.yml && \
    micromamba env create -n diplom -f /tmp/environment.clean.yml && \
    micromamba clean --all --yes

# ======== Pip installs (как обсуждали) ========
RUN micromamba run -n diplom pip install \
      "protobuf==4.25.3" \
      "tensorboard==2.19.0" "tensorboard-data-server==0.7.2" "tensorboard-plugin-wit==1.8.1" \
      "tensorflow==2.19.0" "tensorflow-estimator==2.10.0" \
      "tensorflow-io-gcs-filesystem==0.31.0" \
      "tf-keras==2.19.0"

RUN micromamba run -n diplom pip install \
      opencv-python==4.11.0.86 \
      opencv-contrib-python==4.11.0.86 \
      facenet-pytorch==2.6.0 \
      dlib==19.24.8 \
      albumentations==2.0.8 \
      scipy==1.15.3 \
      fastapi==0.115.12 \
      "starlette>=0.40,<0.47" \
      "uvicorn[standard]==0.30.6" \
      "websockets>=11,<13" \
      "wsproto>=1.2,<1.3" \
      requests==2.32.5 beautifulsoup4==4.13.5 tqdm==4.67.1 yacs==0.1.8 \
      pycocotools==2.0.10 \
      onnx==1.17.0 onnxruntime==1.22.0 \
      python-multipart==0.0.20 retina-face==0.0.17 prettytable==3.16.0 \
      rich==14.1.0 sentencepiece==0.2.1 simsimd==6.5.3 stringzilla==4.0.13 \
      svgwrite==1.4.3 sympy==1.14.0 \
      pydantic==2.11.9 pydantic-core==2.33.2 pydantic-settings==2.10.1 \
      python-dotenv==1.1.1 typing-extensions==4.15.0 typing-inspection==0.4.1 \
      httpx==0.28.1 httpcore==1.0.9 httptools==0.6.4 anyio==4.10.0 idna==3.10 \
      charset-normalizer==3.4.3 psutil==7.1.0 wcwidth==0.2.13 werkzeug==3.1.3 wrapt==1.17.3 && \
    micromamba run -n diplom pip install \
      torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu

RUN micromamba run -n diplom pip install mediapipe==0.10.21

RUN micromamba run -n diplom pip uninstall -y tensorflow-estimator protobuf && \
    micromamba run -n diplom pip install "protobuf>=5.28,<7" && \
    micromamba run -n diplom pip install -U "tensorflow==2.20.0" "tf-keras==2.20.1" "tensorboard==2.20.0"

RUN micromamba run -n diplom pip install face_recognition==1.3.0
EXPOSE 8000


CMD ["uvicorn","gateway:app","--host","0.0.0.0","--port","8000"]

