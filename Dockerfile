ARG NVIDEA_CUDA_BASE_IMAGE
FROM ${NVIDEA_CUDA_BASE_IMAGE} as nvidea-base

# Cog python distribution file downloaded from r8.im/xinntao/gfpgan@sha256:fec8c51c3017f1afbf8b0eed7f4ba51158e49b527b92aea91833aed32559e2ef
COPY cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl

# Install from the file https://www.mssqltips.com/sqlservertip/6802/create-wheel-file-python-package-distribute-custom-code/
RUN pip install /tmp/cog-0.0.1.dev-py3-none-any.whl

# Just a homedir for the modeles
WORKDIR /marcellodesales/replicated/cog/ml-app

# Just getting the yq binary
FROM mikefarah/yq:4.13.2 AS yq-binary

# From the base built above
FROM nvidea-base

# GEtting the binary yq needed
COPY --from=yq-binary /usr/bin/yq /usr/local/bin/yq

# Adding the jq as well
RUN curl -o /usr/local/bin/jq http://stedolan.github.io/jq/download/linux64/jq && \
  chmod +x /usr/local/bin/jq

ONBUILD COPY cog.yaml .

# Install the section .build.system_packages from the cog.yaml config
ONBUILD RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.system_packages[]' | sed -r 's/^([^,]*)(,?)$/ \1 \2/' | tr -d '\n' > cog.pkgs && \
    echo "Installing the system packages: $(cat cog.pkgs)"
ONBUILD RUN apt-get update -qq && apt-get install -qqy $(cat cog.pkgs) && \
    rm -rf /var/lib/apt/lists/* # buildkit 85.8MB buildkit.dockerfile.v0

# Install the section .build.python_packages from the cog.yaml config
ONBUILD RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.python_packages[]' | sed -r 's/^([^,]*)(,?)$/\1 \2/' | tr -d '\n' > cog.python-pkgs && \
    echo "Installing the python packages: $(cat cog.python-pkgs)"
ONBUILD RUN pip install -f https://download.pytorch.org/whl/torch_stable.html $(cat cog.python-pkgs)

# Install the section .build.pre_install from the cog.yaml config
ONBUILD RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.pre_install[]' > cog.pre-inst && \
    echo "Installing the pre-install packages: $(cat cog.pre-inst)"
ONBUILD RUN sh cog.pre-inst

ONBUILD WORKDIR /src

# This is the interface with cog, the predict.py file
ONBUILD COPY predict.py .

# Just copy everything else later
ONBUILD COPY . .

# The requirements should be moved from the requirements.txt to the cog file
#ONBUILD RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-m", "cog.server.http"]
