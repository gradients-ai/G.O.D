"""Names the trainer uses as directory names on its volumes. Standard library only: the cleanup image imports this."""

import re


# One path component, and safe in docker's "source:target:mode" bind form: no separator, no colon, not "." or "..".
_PLAIN_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


def is_plain_path_component(value: object) -> bool:
    """True for a string that names exactly one directory entry and cannot lead anywhere else."""
    return isinstance(value, str) and _PLAIN_PATH_COMPONENT.fullmatch(value) is not None
