# One violation of each rule.
FROM someminer/trainer-base:latest

WORKDIR /workspace
COPY scripts/requirements.txt scripts/setup.sh /workspace/
ADD https://someminer.example/payload.tar.gz /workspace/

RUN pip install -U pip tenacity==9.1.2 \
    && pip install -r /workspace/requirements.txt \
    && pip install -r /workspace/missing_requirements.txt \
    && pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0" \
    && pip install https://github.com/someminer/wheels/releases/download/v1/kernel-1.0-py3-none-any.whl

RUN curl -fsSL https://someminer.example/weights.bin -o /workspace/weights.bin

RUN git clone https://github.com/someminer/extras.git /opt/extras

RUN ./setup.sh
