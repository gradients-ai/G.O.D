from pydantic import BaseModel
from pydantic import Field


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""

    reward_id: str | None = Field(None, description="UUID of the reward function in the database")
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            '"""Reward function that favors shorter, more concise answers."""',
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]",
        ],
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None
    is_manual: bool | None = None
