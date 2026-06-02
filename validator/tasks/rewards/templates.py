"""
Reward function template registry.

Each template defines a metric and how to score it. At task creation time,
parameters are sampled and baked into a concrete function string that all
miners receive identically.

Templates are the SINGLE source of truth for synthetic reward functions.
Static DB functions may coexist but templates are the primary source.

Three scoring modes:
    - directional:   sign * metric(completion)
    - target:        -abs(target - metric(completion))
    - binary:        1.0 if predicate(completion) else 0.0

Four template shapes (by how the metric is computed):
    1. Pure Python per-completion expression
    2. textstat per-completion function call
    3. langcheck batched model prediction
    4. Regex pattern match
"""

from __future__ import annotations

import random
import textwrap
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ScoringMode(str, Enum):
    DIRECTIONAL = "directional"
    TARGET = "target"
    BINARY = "binary"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"


class ParamSpec(BaseModel):
    """Schema for a single template parameter."""

    param_type: ParamType
    min_val: float | None = None
    max_val: float | None = None
    choices: list[Any] | None = None

    def sample(self, rng: random.Random) -> Any:
        if self.param_type == ParamType.INT:
            return rng.randint(int(self.min_val), int(self.max_val))
        if self.param_type == ParamType.FLOAT:
            return round(rng.uniform(self.min_val, self.max_val), 4)
        if self.param_type == ParamType.CHOICE:
            return rng.choice(self.choices)
        raise ValueError(f"Unknown param type: {self.param_type}")


class RewardTemplate(BaseModel):
    """A parameterized reward function template.

    The ``code_template`` uses Python str.format() placeholders that correspond
    to keys in ``params``.  At instantiation time each param is sampled and
    substituted to produce a concrete, self-contained function string.
    """

    name: str = Field(..., description="Unique template identifier, e.g. 'word_count'")
    description: str = Field(..., description="Human-readable description of what this template measures")
    scoring_mode: ScoringMode
    code_template: str = Field(..., description="Python function string with {param} placeholders")
    params: dict[str, ParamSpec] = Field(default_factory=dict)

    def instantiate(self, rng: random.Random | None = None) -> str:
        """Sample all parameters and return a concrete function string."""
        rng = rng or random.Random()
        values = {key: spec.sample(rng) for key, spec in self.params.items()}
        return self.code_template.format(**values)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

# 1. Pure-python per-completion metrics
# ---------------------------------------------------------------------------

CHAR_COUNT = RewardTemplate(
    name="char_count",
    description="Reward completions close to a target character count",
    scoring_mode=ScoringMode.TARGET,
    code_template=textwrap.dedent("""\
        def reward_char_count(completions, **kwargs):
            target = {target}
            return [-abs(target - len(c)) for c in completions]
    """),
    params={
        "target": ParamSpec(param_type=ParamType.INT, min_val=100, max_val=1500),
    },
)

WORD_COUNT = RewardTemplate(
    name="word_count",
    description="Reward completions close to a target word count",
    scoring_mode=ScoringMode.TARGET,
    code_template=textwrap.dedent("""\
        def reward_word_count(completions, **kwargs):
            target = {target}
            return [-abs(target - len(c.split())) for c in completions]
    """),
    params={
        "target": ParamSpec(param_type=ParamType.INT, min_val=15, max_val=300),
    },
)

SENTENCE_COUNT = RewardTemplate(
    name="sentence_count",
    description="Reward completions close to a target number of sentences",
    scoring_mode=ScoringMode.TARGET,
    code_template=textwrap.dedent("""\
        def reward_sentence_count(completions, **kwargs):
            import re
            target = {target}
            scores = []
            for c in completions:
                sentences = [s for s in re.split(r'[.!?]+', c) if s.strip()]
                scores.append(-abs(target - len(sentences)))
            return scores
    """),
    params={
        "target": ParamSpec(param_type=ParamType.INT, min_val=1, max_val=15),
    },
)

UNIQUE_WORD_RATIO = RewardTemplate(
    name="unique_word_ratio",
    description="Reward completions based on lexical diversity (unique words / total words)",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_unique_word_ratio(completions, **kwargs):
            sign = {sign}
            scores = []
            for c in completions:
                words = c.split()
                ratio = len(set(words)) / len(words) if words else 0.0
                scores.append(sign * ratio)
            return scores
    """),
    params={
        "sign": ParamSpec(param_type=ParamType.CHOICE, choices=[1, -1]),
    },
)

COMPLETION_LENGTH = RewardTemplate(
    name="completion_length",
    description="Reward longer or shorter completions by character count",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_completion_length(completions, **kwargs):
            sign = {sign}
            return [sign * float(len(c)) for c in completions]
    """),
    params={
        "sign": ParamSpec(param_type=ParamType.CHOICE, choices=[1, -1]),
    },
)

KEYWORD_PRESENCE = RewardTemplate(
    name="keyword_presence",
    description="Reward presence of reasoning and analytical keywords",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_keyword_presence(completions, **kwargs):
            keywords = {keywords}
            return [
                sum(1 for kw in keywords if kw in c.lower())
                for c in completions
            ]
    """),
    params={
        "keywords": ParamSpec(
            param_type=ParamType.CHOICE,
            choices=[
                [
                    "because", "therefore", "thus", "hence", "consequently",
                    "however", "nevertheless", "although", "despite", "whereas",
                    "example", "instance", "specifically", "particularly",
                    "furthermore", "moreover", "additionally",
                    "analyze", "evaluate", "consider", "examine", "determine",
                ],
                [
                    "first", "second", "third", "finally", "in conclusion",
                    "on the other hand", "in contrast", "similarly",
                    "for instance", "such as", "in particular",
                    "as a result", "given that", "assuming that",
                ],
                [
                    "step", "process", "method", "approach", "technique",
                    "solution", "algorithm", "procedure", "strategy",
                    "implement", "execute", "apply", "design", "optimize",
                ],
            ],
        ),
    },
)


# 2. textstat per-completion metrics
# ---------------------------------------------------------------------------

TEXTSTAT_DIRECTIONAL = RewardTemplate(
    name="textstat_directional",
    description="Maximize or minimize a textstat readability/complexity metric",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_textstat_{metric}(completions, **kwargs):
            import textstat
            sign = {sign}
            return [sign * textstat.{metric}(c) for c in completions]
    """),
    params={
        "metric": ParamSpec(
            param_type=ParamType.CHOICE,
            choices=[
                "words_per_sentence",
                "avg_character_per_word",
                "avg_syllables_per_word",
                "flesch_reading_ease",
            ],
        ),
        "sign": ParamSpec(param_type=ParamType.CHOICE, choices=[1, -1]),
    },
)

_TEXTSTAT_TARGET_CODE = textwrap.dedent("""\
    def reward_textstat_{metric}_target(completions, **kwargs):
        import textstat
        target = {target}
        scale = {scale}
        return [
            -(abs(textstat.{metric}(c) - target) / scale)
            for c in completions
        ]
""")

GRADE_LEVEL_TARGET = RewardTemplate(
    name="grade_level_target",
    description="Reward completions matching a target Flesch-Kincaid grade level (1-16)",
    scoring_mode=ScoringMode.TARGET,
    code_template=_TEXTSTAT_TARGET_CODE,
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["flesch_kincaid_grade"]),
        "target": ParamSpec(param_type=ParamType.FLOAT, min_val=3.0, max_val=16.0),
        "scale": ParamSpec(param_type=ParamType.FLOAT, min_val=3.0, max_val=8.0),
    },
)

READING_EASE_TARGET = RewardTemplate(
    name="reading_ease_target",
    description="Reward completions matching a target Flesch reading ease score (0-100)",
    scoring_mode=ScoringMode.TARGET,
    code_template=_TEXTSTAT_TARGET_CODE,
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["flesch_reading_ease"]),
        "target": ParamSpec(param_type=ParamType.FLOAT, min_val=20.0, max_val=90.0),
        "scale": ParamSpec(param_type=ParamType.FLOAT, min_val=10.0, max_val=30.0),
    },
)

WORDS_PER_SENTENCE_TARGET = RewardTemplate(
    name="words_per_sentence_target",
    description="Reward completions matching a target average words per sentence (5-40)",
    scoring_mode=ScoringMode.TARGET,
    code_template=_TEXTSTAT_TARGET_CODE,
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["words_per_sentence"]),
        "target": ParamSpec(param_type=ParamType.FLOAT, min_val=5.0, max_val=35.0),
        "scale": ParamSpec(param_type=ParamType.FLOAT, min_val=5.0, max_val=15.0),
    },
)

CHARS_PER_WORD_TARGET = RewardTemplate(
    name="chars_per_word_target",
    description="Reward completions matching a target average characters per word (3-8)",
    scoring_mode=ScoringMode.TARGET,
    code_template=_TEXTSTAT_TARGET_CODE,
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["avg_character_per_word"]),
        "target": ParamSpec(param_type=ParamType.FLOAT, min_val=3.0, max_val=7.0),
        "scale": ParamSpec(param_type=ParamType.FLOAT, min_val=1.0, max_val=3.0),
    },
)

SYLLABLES_PER_WORD_TARGET = RewardTemplate(
    name="syllables_per_word_target",
    description="Reward completions matching a target average syllables per word (1-4)",
    scoring_mode=ScoringMode.TARGET,
    code_template=_TEXTSTAT_TARGET_CODE,
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["avg_syllables_per_word"]),
        "target": ParamSpec(param_type=ParamType.FLOAT, min_val=1.0, max_val=3.5),
        "scale": ParamSpec(param_type=ParamType.FLOAT, min_val=0.5, max_val=2.0),
    },
)

DIFFICULT_WORD_RATIO = RewardTemplate(
    name="difficult_word_ratio",
    description="Reward based on the ratio of difficult words in the completion",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_difficult_word_ratio(completions, **kwargs):
            import textstat
            sign = {sign}
            scores = []
            for c in completions:
                words = c.split()
                if not words:
                    scores.append(0.0)
                else:
                    scores.append(sign * textstat.difficult_words(c) / len(words))
            return scores
    """),
    params={
        "sign": ParamSpec(param_type=ParamType.CHOICE, choices=[1, -1]),
    },
)


# 3. langcheck batched model metrics
# ---------------------------------------------------------------------------

LANGCHECK_SCORE = RewardTemplate(
    name="langcheck_score",
    description="Score completions on a langcheck quality metric",
    scoring_mode=ScoringMode.DIRECTIONAL,
    code_template=textwrap.dedent("""\
        def reward_langcheck_{metric}(completions, **kwargs):
            import langcheck
            scores = langcheck.metrics.{metric}(completions)
            sign = {sign}
            return [sign * s for s in scores.metric_values]
    """),
    params={
        "metric": ParamSpec(param_type=ParamType.CHOICE, choices=["sentiment", "fluency"]),
        "sign": ParamSpec(param_type=ParamType.CHOICE, choices=[1, -1]),
    },
)


# 4. Regex pattern match
# ---------------------------------------------------------------------------

REGEX_FORMAT = RewardTemplate(
    name="regex_format",
    description="Binary reward for completions matching a regex pattern",
    scoring_mode=ScoringMode.BINARY,
    code_template=textwrap.dedent("""\
        def reward_regex_format(completions, **kwargs):
            import re
            pattern = r"{pattern}"
            return [1.0 if re.search(pattern, c) else 0.0 for c in completions]
    """),
    params={
        "pattern": ParamSpec(
            param_type=ParamType.CHOICE,
            choices=[
                r"^<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>",
                r"^\d+\.\s",
                r"\b(in conclusion|to summarize|in summary)\b",
                r"```[\s\S]*?```",
            ],
        ),
    },
)


# ---------------------------------------------------------------------------
# Groups — templates that measure the same dimension live together.
# Sampling picks one group, then one template within it.  Two templates
# from the same group never appear on the same task.
# ---------------------------------------------------------------------------

TEMPLATE_GROUPS: dict[str, list[RewardTemplate]] = {
    "length": [CHAR_COUNT, WORD_COUNT, COMPLETION_LENGTH],
    "readability": [READING_EASE_TARGET, GRADE_LEVEL_TARGET, TEXTSTAT_DIRECTIONAL],
    "word_complexity": [CHARS_PER_WORD_TARGET, SYLLABLES_PER_WORD_TARGET, DIFFICULT_WORD_RATIO],
    "sentence_structure": [SENTENCE_COUNT, WORDS_PER_SENTENCE_TARGET],
    "vocabulary": [UNIQUE_WORD_RATIO, KEYWORD_PRESENCE],
    "quality": [LANGCHECK_SCORE],
    "format": [REGEX_FORMAT],
}

# Flat list for convenience
TEMPLATE_REGISTRY: list[RewardTemplate] = [
    t for group in TEMPLATE_GROUPS.values() for t in group
]


def sample_template_groups(
    n: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample n distinct groups and instantiate one template per group.

    Returns a list of concrete Python function strings ready to store on a task.
    Two functions from the same group never appear together, preventing
    contradictory training signals.
    """
    rng = rng or random.Random()

    group_names = list(TEMPLATE_GROUPS.keys())
    n = min(n, len(group_names))
    selected_groups = rng.sample(group_names, n)

    return [
        rng.choice(TEMPLATE_GROUPS[group_name]).instantiate(rng)
        for group_name in selected_groups
    ]
