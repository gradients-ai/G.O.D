"""The tokenized unit every text evaluator scores.

One shape for instruct, completion and chat rows so the dataset builder, the collator and the eval-set
fingerprint never care which labelling rule produced an example. The validator rejects labels that are not
either masked or the input id at the same position: a label pointing at a different token would make the loss
measure something other than "predict the next token of this sequence", and that is never intended.
"""

from pydantic import BaseModel
from pydantic import model_validator


IGNORE_INDEX = -100


class TokenizedExample(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

    @model_validator(mode="after")
    def _labels_follow_input_ids(self) -> "TokenizedExample":
        if not (len(self.input_ids) == len(self.attention_mask) == len(self.labels)):
            raise ValueError(
                f"length mismatch: input_ids={len(self.input_ids)} attention_mask={len(self.attention_mask)} "
                f"labels={len(self.labels)}"
            )
        for position, (token_id, label) in enumerate(zip(self.input_ids, self.labels)):
            if label != IGNORE_INDEX and label != token_id:
                raise ValueError(f"label {label} at position {position} is neither {IGNORE_INDEX} nor the input id {token_id}")
        return self

    def n_scored(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)
