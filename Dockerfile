ARG NVIDEA_CUDA_BASE_IMAGE
FROM ${NVIDEA_CUDA_BASE_IMAGE}

# Cog python distribution file downloaded from r8.im/xinntao/gfpgan@sha256:fec8c51c3017f1afbf8b0eed7f4ba51158e49b527b92aea91833aed32559e2ef
COPY cog-0.0.1.dev-py3-none-any.whl /tmp/cog.whl

RUN pip install /tmp/cog.whl

WORKDIR /marcellodesales/replicated/cog/app

CMD ["python3", "--version"]