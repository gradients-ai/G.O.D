"""Prints the quant eval GPU as QuantEvalGpu JSON. Run in the quant image as a subprocess of
evaluators/quant.py, which stays free of torch."""

import torch

from core.models.quant_models import QuantEvalGpu


def read_gpu() -> QuantEvalGpu:
    props = torch.cuda.get_device_properties(0)
    return QuantEvalGpu(name=props.name, sm_count=props.multi_processor_count)


if __name__ == "__main__":
    print(read_gpu().model_dump_json())
