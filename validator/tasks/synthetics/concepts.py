"""Locked-concept synthetic image tasks.

Every task teaches one concept a base model cannot know from its name: a product, a design system, a brand identity,
a campaign look, an invented art style, a mascot drawn in an invented style, or a named edit rule.

What makes these tasks hard, and what the generator enforces:
- One identity, locked by a detailed spec and (where the concept is visual) one reference image that every training
  and test image is rendered from, so the concept is consistent across the dataset.
- Captions are short (trigger + context). The detailed spec goes to the image generator only, so a LoRA has to learn
  the concept's look instead of reading it from the caption.
- Training and held-out test contexts come from different places (studio vs in use, phone vs billboard, known vs
  unseen object types, two concepts trained apart and tested together). The held-out pairs are generated last; the
  task's last IMAGE_SYNTH_TEST_PAIRS pairs are its test split (see train_test_split_image).
- Diversity: randomized creative briefs per kind, a random starting letter for invented names, a ban on overused name
  fragments, and rejection of concepts too similar to recent ones.
"""

import asyncio
import json
import random
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import validator.tasks.synthetics.constants as synth_cst
from core.logging import get_logger


logger = get_logger(__name__)

with open(Path(__file__).with_name("image_synth_concepts.json"), "r") as f:
    CONCEPT_TEMPLATES = json.load(f)

T2I_KINDS = ("product", "design", "logo", "social", "style", "compose")
RECENT_CONCEPTS: dict[str, deque[str]] = {
    kind: deque(maxlen=synth_cst.IMAGE_SYNTH_RECENT_CONCEPT_LIMIT) for kind in (*T2I_KINDS, "edit")
}
OVERUSED_NAME_FRAGMENTS = (
    "aether", "aura", "lumi", "lumo", "luna", "nova", "zen", "flux", "neo", "echo", "vibe", "sol", "terra",
    "nexus", "orbit", "pulse", "astra", "verde", "kairo", "zephyr",
)
SUBJECT_DOMAINS = (
    "landscapes and weather", "city streets and architecture", "still life and tableware", "wild animals",
    "people at work", "vehicles and machines", "food and markets", "domestic interiors", "plants and gardens",
    "sports and games", "musical instruments", "the sea and boats", "night scenes and lights", "festivals and crowds",
)
OBJECT_GROUPS = (
    "kitchenware", "musical instruments", "footwear", "small electronics", "furniture", "hand tools", "sports gear",
    "toys and games", "street furniture", "garden tools", "office supplies", "bathroom items", "luggage and bags",
    "lighting fixtures", "clocks and timepieces", "bicycles and scooters",
)


@dataclass(frozen=True)
class ConceptSpec:
    kind: str
    trigger: str
    identity_spec: str
    train_contexts: list[str]
    test_contexts: list[str]
    reference_prompt: str = ""
    # compose only: the second concept, trained separately from the mascot and combined only in the test split
    style_name: str = ""
    style_spec: str = ""
    style_subjects: list[str] = field(default_factory=list)


def _pick(options, k=1):
    return random.sample(options, k) if k > 1 else random.choice(options)


def creative_brief(kind: str) -> dict[str, str]:
    """Randomized direction per kind; every value lands in the LLM request, so consecutive tasks differ."""
    letter = random.choice("BCDFGHJKLMPRSTVWZ" + "AEIOU")
    brief = {"letter": letter, "banned_fragments": ", ".join(OVERUSED_NAME_FRAGMENTS)}
    if kind == "product":
        brief["brief"] = (
            f"Domain: {_pick(CONCEPT_TEMPLATES['briefs']['product_domains'])}. "
            f"Materials: {' and '.join(_pick(CONCEPT_TEMPLATES['briefs']['materials'], 2))}. "
            f"Form: {_pick(CONCEPT_TEMPLATES['briefs']['forms'])}."
        )
    elif kind == "design":
        brief["surface"] = _pick(("mobile app", "web app"))
        brief["brief"] = (
            f"Product area: {_pick(CONCEPT_TEMPLATES['briefs']['app_domains'])}. Visual direction: "
            f"{_pick(CONCEPT_TEMPLATES['briefs']['visual_directions'])}."
        )
    elif kind == "logo":
        brief["brief"] = (
            f"Industry: {_pick(CONCEPT_TEMPLATES['briefs']['industries'])}. Mark type: "
            f"{_pick(CONCEPT_TEMPLATES['briefs']['mark_types'])}. Palette: {_pick(CONCEPT_TEMPLATES['briefs']['palettes'])}."
        )
    elif kind == "social":
        brief["brief"] = (
            f"Campaign: {_pick(CONCEPT_TEMPLATES['briefs']['campaigns'])}. Layout motif: "
            f"{_pick(CONCEPT_TEMPLATES['briefs']['layout_motifs'])}. Palette: {_pick(CONCEPT_TEMPLATES['briefs']['palettes'])}."
        )
    elif kind in ("style", "compose"):
        train_domains = _pick(SUBJECT_DOMAINS, 3)
        test_domains = _pick([d for d in SUBJECT_DOMAINS if d not in train_domains], 3)
        brief["train_domains"], brief["test_domains"] = "; ".join(train_domains), "; ".join(test_domains)
        brief["brief"] = (
            f"Medium to start from: {_pick(CONCEPT_TEMPLATES['briefs']['media'])}. Twist: "
            f"{_pick(CONCEPT_TEMPLATES['briefs']['style_twists'])}."
        )
        if kind == "compose":
            brief["mascot"] = _pick(CONCEPT_TEMPLATES["briefs"]["mascot_materials"])
    elif kind == "edit":
        train_groups = _pick(OBJECT_GROUPS, 4)
        test_groups = _pick([g for g in OBJECT_GROUPS if g not in train_groups], 3)
        brief["train_groups"], brief["test_groups"] = "; ".join(train_groups), "; ".join(test_groups)
        brief["brief"] = f"Transformation family: {_pick(CONCEPT_TEMPLATES['briefs']['edit_families'])}."
    return brief


def concept_request(kind: str, num_train: int, num_test: int, brief: dict[str, str], avoid: list[str]) -> str:
    counts = {"num_mascot": num_train - num_train // 2, "num_style": num_train // 2} if kind == "compose" else {}
    body = CONCEPT_TEMPLATES[kind].format(num_train=num_train, num_test=num_test, **counts, **brief)
    common = CONCEPT_TEMPLATES["common"].format(
        avoid="; ".join(avoid) or "none yet", num_train=num_train, num_test=num_test, **brief
    )
    return body + "\n\n" + common


def _clean_list(result: dict, key: str, count: int) -> list[str]:
    values = result.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Concept response missing {key} list")
    clean = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if len(clean) < count:
        raise ValueError(f"Concept response has {len(clean)} {key}, expected {count}")
    return clean[:count]


def _clean_text(result: dict, key: str) -> str:
    value = result.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Concept response missing {key}")
    return value.strip()


def parse_concept(kind: str, result: dict, num_train: int, num_test: int) -> ConceptSpec:
    """Validate the LLM's JSON for one concept. Compose splits the training budget between its two concepts."""
    trigger = _clean_text(result, "trigger")
    if any(fragment in trigger.casefold() for fragment in OVERUSED_NAME_FRAGMENTS):
        raise ValueError(f"Concept name '{trigger}' uses an overused fragment")
    if kind == "social":  # without its own headline a post copies the reference's headline and subject
        missing = [c for c in result.get("train_contexts", []) + result.get("test_contexts", []) if '"' not in c and "'" not in c]
        if missing:
            raise ValueError(f"Social contexts without a quoted headline: {missing[:2]}")
    if kind == "compose":
        n_mascot = num_train - num_train // 2
        return ConceptSpec(
            kind=kind,
            trigger=trigger,
            identity_spec=_clean_text(result, "identity_spec"),
            reference_prompt=_clean_text(result, "reference_prompt"),
            train_contexts=_clean_list(result, "train_contexts", n_mascot),
            test_contexts=_clean_list(result, "test_contexts", num_test),
            style_name=_clean_text(result, "style_name"),
            style_spec=_clean_text(result, "style_spec"),
            style_subjects=_clean_list(result, "style_subjects", num_train // 2),
        )
    return ConceptSpec(
        kind=kind,
        trigger=trigger,
        identity_spec=_clean_text(result, "identity_spec"),
        reference_prompt=_clean_text(result, "reference_prompt") if kind != "style" else "",
        train_contexts=_clean_list(result, "train_contexts", num_train),
        test_contexts=_clean_list(result, "test_contexts", num_test),
    )


def similar_to_recent(kind: str, text: str, similarity) -> bool:
    return any(similarity(text, previous) >= synth_cst.IMAGE_SYNTH_CONCEPT_SIMILARITY_LIMIT for previous in RECENT_CONCEPTS[kind])


def _with_trigger(trigger: str, context: str, sep: str = " ") -> str:
    """Caption text: the context, prefixed with the trigger unless the context already names it."""
    return context if trigger.casefold() in context.casefold() else f"{trigger}{sep}{context}"


def render_jobs(spec: ConceptSpec, reference_url: str | None) -> list[tuple[str, str | None, str, str]]:
    """(model, reference_url, generation prompt, caption) per image: training first, held-out test last.

    The generation prompt carries the full identity spec; the caption is only the trigger plus the context.
    """
    keep = CONCEPT_TEMPLATES["keep"][spec.kind].format(trigger=spec.trigger, spec=spec.identity_spec)
    edit_model, t2i_model = synth_cst.FAL_AVATAR_MODEL, synth_cst.FAL_STYLE_MODEL_GPT_IMAGE_2
    if spec.kind == "style":
        return [(t2i_model, None, f"{ctx}. {keep}", f"{ctx}, {spec.trigger}") for ctx in spec.train_contexts + spec.test_contexts]
    if spec.kind == "compose":
        style = f"in the {spec.style_name} style: {spec.style_spec}"
        jobs = [(edit_model, reference_url, f"{keep} {_with_trigger(spec.trigger, ctx)}. A real photograph.",
                 f"{_with_trigger(spec.trigger, ctx)}, photo") for ctx in spec.train_contexts]
        jobs += [(t2i_model, None, f"{subject}, {style}.", f"{subject}, {spec.style_name}") for subject in spec.style_subjects]
        jobs += [(edit_model, reference_url,
                  f"Draw {spec.trigger} from the reference image, keeping its exact shape, colours and details "
                  f"({spec.identity_spec}), {ctx}, {style}. An artwork, not a photo.",
                  f"{_with_trigger(spec.trigger, ctx)}, {spec.style_name}") for ctx in spec.test_contexts]
        return jobs
    sep = " " if spec.kind == "product" else ", "
    captions = [_with_trigger(spec.trigger, ctx, sep) for ctx in spec.train_contexts + spec.test_contexts]
    return [(edit_model, reference_url, f"{keep} {caption}.", caption) for caption in captions]


def qa_request(spec: ConceptSpec, context: str) -> str:
    noun, extra = CONCEPT_TEMPLATES["qa_nouns"][spec.kind]
    return CONCEPT_TEMPLATES["qa"].format(trigger=spec.trigger, noun=noun, spec=spec.identity_spec, context=context, extra=extra)


def reference_prompt(spec: ConceptSpec) -> str:
    """The reference image gets the full identity spec, so it cannot drift from what every caption refers to."""
    return f"{spec.reference_prompt} {spec.identity_spec}"


def reference_qa_request(spec: ConceptSpec) -> str:
    noun, _ = CONCEPT_TEMPLATES["qa_nouns"][spec.kind]
    return CONCEPT_TEMPLATES["qa_reference"].format(trigger=spec.trigger, noun=noun, spec=spec.identity_spec)


def qa_passed(result: dict) -> bool:
    return result.get("same") is True and result.get("visible") is True


@dataclass(frozen=True)
class RuleEditSpec:
    goal_name: str
    rule_spec: str
    train_subjects: list[str]
    test_subjects: list[str]

    @property
    def instruction(self) -> str:
        # The caption names the rule only; the rule's look must be learned from the pairs.
        return f"Apply the {self.goal_name}."


def parse_rule_edit(result: dict, num_train: int, num_test: int) -> RuleEditSpec:
    name = _clean_text(result, "goal_name")
    if any(fragment in name.casefold() for fragment in OVERUSED_NAME_FRAGMENTS):
        raise ValueError(f"Edit rule name '{name}' uses an overused fragment")
    return RuleEditSpec(
        goal_name=name,
        rule_spec=_clean_text(result, "rule_spec"),
        train_subjects=_clean_list(result, "train_subjects", num_train),
        test_subjects=_clean_list(result, "test_subjects", num_test),
    )


def rule_edit_source_prompt(subject: str) -> str:
    return CONCEPT_TEMPLATES["edit_source"].format(subject=subject)


async def gather_limited(coroutines, limit: int = synth_cst.FAL_IMAGE_GENERATION_CONCURRENCY):
    semaphore = asyncio.Semaphore(limit)

    async def run(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(run(c) for c in coroutines))
