# Use Ubuntu 20.04 as the base image
# Create a temp image that has build tools that we can use to build wheel
# files for dependencies only available as source.
FROM docker-registry.qualcomm.com/library/ubuntu:20.04

# Update the package lists and install required packages
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    python3.12 \
    python3.12-venv \
    python3-pip

# pip recognizes this variable
ENV PIP_CACHE_DIR=/var/cache/pip
WORKDIR /app

# Sample command to register and clone the repository
# Clone the GitHub repository
RUN git config --global user.email none@none.com && \
    git config --global user.name none

RUN mkdir -p /app/qefficient-library
COPY . /app/qefficient-library

# Create Virtual Env for the docker image
RUN python3.12 -m venv /app/llm_env
RUN . /app/llm_env/bin/activate
WORKDIR /app/qefficient-library

# Install the required Python packages

RUN pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu --no-deps
RUN pip install datasets==2.17.0 fsspec==2023.10.0 multidict==6.0.5 sentencepiece --no-deps

RUN python3.12 -m pip install .
WORKDIR /app/qefficient-library

# Set the environment variable for the model card name and token ID
ENV HF_HOME = "/app/qefficient-library/docs"
ENV MODEL_NAME = ""
ENV CACHE_DIR = ""
ENV TOKEN_ID = ""

# Print a success message
CMD ["echo", "qefficient-transformers repository cloned and setup installed inside Docker image."]
CMD ["echo", "Starting the Model Download and Export to Onnx Stage for QEff."]
CMD python3.12 -m QEfficient.cloud.export --model-name "$MODEL_NAME"

# Example usage:
# docker build -t qefficient-library .

# Minimum System Requirements Before running docker containers: 
# 1. Clear the tmp space.
# 2. For smaller models, 32GiB RAM is sufficient, but larger LLMs we require good CPU/RAM (Context 7B model would require atleast 64GiB).
# 3. The exact minimum system configuration are tough to decide, since its all function of model parameters.

# docker run -e MODEL_NAME=gpt2 -e TOKEN_ID=<your-token-id> qefficient-library
