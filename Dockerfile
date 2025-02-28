# FROM pytorch/torchserve:latest-gpu
# USER root
# RUN apt-get update && apt-get install -y espeak-ng git && rm -rf /var/lib/apt/lists/*
# RUN --mount=type=bind,source=requirements.txt,target=/app/requirements.txt pip install -r /app/requirements.txt && rm -rf /root/.cache/pip && python -m nltk.downloader punkt_tab
# ENTRYPOINT [ "torchserve", "--start" ]

FROM pytorch/torchserve:latest-gpu

USER root

# Update package list, install system dependencies, and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y python3-dev build-essential python3-setuptools python3-distutils espeak-ng git && \
    # gcc g++ libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt && rm -rf /root/.cache/pip

# Download the NLTK data non-interactively
RUN python -c "import nltk; nltk.download('punkt')"

# Set the entrypoint to start TorchServe
ENTRYPOINT ["torchserve", "--start"]