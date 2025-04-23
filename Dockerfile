FROM ghcr.io/intertwin-eu/itwinai:torch-slim-latest

# Set working directory
WORKDIR /app

# Remove itwinai data under /app
# RUN rm -rf tests src pyproject.toml

# Copy your scripts
COPY configuration_files/ configuration_files/
COPY src/ src/
COPY pyproject.toml pyproject.toml

# Install dependencies
RUN pip install uv
RUN uv pip install --no-cache-dir .

