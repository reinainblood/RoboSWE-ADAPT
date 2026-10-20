FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir numpy==2.2.5 scipy==1.15.3 Pillow==11.2.1 \
    && mkdir -p /workspace && chown 1000:1000 /workspace
COPY runner.py robot.py archive.py /opt/roboswe/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/opt/roboswe:/workspace
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT ["python", "-I", "-B", "-u", "/opt/roboswe/runner.py"]
