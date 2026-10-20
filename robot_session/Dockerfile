FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYNPUT_BACKEND=dummy \
    NUMBA_CACHE_DIR=/tmp/numba

ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libosmesa6 libgl1 libegl1 libglib2.0-0 libglfw3 \
    libxrender1 libxext6 libsm6 libusb-1.0-0 libhidapi-hidraw0 \
    && rm -rf /var/lib/apt/lists/*

# Simulation-only runtime: training / dataset-conversion extras are not needed.
COPY robot_session/requirements.txt /opt/session-requirements.txt
RUN pip install --no-cache-dir -r /opt/session-requirements.txt && pip check

# Include assets in the immutable image. Fresh replay must not depend on mutable
# host bind mounts or on the agent's writable filesystem.
COPY upstream/robosuite /opt/robosuite
COPY upstream/robocasa /opt/robocasa
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/robosuite -e /opt/robocasa
ENV PYTHONPATH=/app:/opt/robocasa:/opt/robosuite
COPY robot_session /app/robot_session
COPY roboswe /app/roboswe
COPY pairing /app/pairing
RUN mkdir -p /workspace && pip freeze > /app/runtime-packages.txt
WORKDIR /workspace
CMD ["python", "-m", "robot_session.session"]
