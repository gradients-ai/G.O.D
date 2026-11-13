from enum import Enum


class TournamentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    QUANT = "quant"
    ENVIRONMENT = "environment"
