# This is a sample Dockerfile for JupyterLab containers based on itwinai containers

# DO NOT UPGRADE OR DOWNGRADE ALL JUPYTER PACKAGES (E.G., JUPYTERLAB), unless you know what you are doing
# Otherwise you risk of breaking the spawn in JupyterHub

FROM ghcr.io/intertwin-eu/itwinai:jlab-slim-latest

# Set working directory
WORKDIR /app

# Remove itwinai data under /app
RUN rm -rf tests src pyptoject.toml Dockerfile

# Copy application dependencies.
# Remember that you are not root in this container, so you need to
# propagate the correct rights when copying files.
COPY  --chown=${NB_UID} pyproject.toml pyproject.toml
COPY  --chown=${NB_UID} src src

# Install dependencies
RUN pip install --no-cache-dir .

# Copy tests
COPY --chown=${NB_UID} tests tests

# DO NOT SET AN ENTRYPOINT OR A CMD, unless you know what you are doing
# Otherwise you risk of breaking the spawn in JupyterHub