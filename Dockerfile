FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \ 
    libglpk-dev \
    python3-dev \ 
    libcdd-dev \ 
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace
COPY . .

# Install python packages
RUN pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

CMD /bin/sh
