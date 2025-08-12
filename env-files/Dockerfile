FROM ghcr.io/intertwin-eu/itwinai:torch-slim-latest
LABEL org.opencontainers.image.source="https://github.com/interTwin-eu/hython-itwinai-plugin"

# Set working directory
WORKDIR /app

# Remove itwinai data under /app
RUN rm -rf tests src pyproject.toml

# Copy your scripts
COPY src/ src/
COPY pyproject.toml pyproject.toml

# Install dependencies
RUN uv pip install --no-cache-dir .

# Copy tests
COPY tests tests
