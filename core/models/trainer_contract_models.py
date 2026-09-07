from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class GPUType(str, Enum):
    H100 = "H100"
    A100 = "A100"
    A6000 = "A6000"


class GPUInterconnect(str, Enum):
    """Normalized GPU form-factor / interconnect class."""

    SXM = "SXM" 
    PCIE = "PCIe"
    NVL = "NVL"
    UNKNOWN = "unknown"


class GPUInfo(BaseModel):
    gpu_id: int = Field(..., description="GPU ID")
    gpu_type: GPUType = Field(..., description="GPU Type")
    vram_gb: int = Field(..., description="GPU VRAM in GB")
    available: bool = Field(..., description="GPU Availability")
    used_until: datetime | None = Field(default=None, description="GPU Used Until")
    updated_at: datetime | None = Field(default=None, description="When GPU status was last updated")
    product_name: str | None = Field(default=None, description="Raw NVML product name")
    interconnect: GPUInterconnect = Field(
        default=GPUInterconnect.UNKNOWN,
        description="Form factor / interconnect: SXM, PCIe, NVL, or unknown",
    )
    nvlink: bool = Field(default=False, description="True if any NVLink link is enabled on this GPU")
