"""chat_labels.py is one rule shared by the standard eval, the quant eval and the miners' copies: pin its bytes.

A change to the rule must be made upstream first and be copied here verbatim, updating this hash.
"""

import hashlib
from pathlib import Path

from validator.evaluation import chat_labels


# sha256 of validator/evaluation/chat_labels.py at the upstream revision it was copied from.
CHAT_LABELS_SHA256 = "a57149ba905634419e2ef80dfeacf569b72859bc902bc6e9683e7c4b74eadff6"


def test_chat_labels_is_byte_identical_to_the_standard_evals():
    assert hashlib.sha256(Path(chat_labels.__file__).read_bytes()).hexdigest() == CHAT_LABELS_SHA256
