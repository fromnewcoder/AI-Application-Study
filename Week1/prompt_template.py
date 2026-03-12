"""
prompt_template.py
==================
Prompt Engineering II — Advanced
A reusable, composable PromptTemplate class that encodes:
  • Chain-of-Thought (CoT) scaffolding
  • XML structuring for inputs and outputs
  • Negative examples (counter-demonstrations)
  • Few-shot example management
  • System / user message separation
  • Template variable interpolation with validation

Author:  Prompt Engineering Series — Module II
Python:  3.9+
Deps:    None (stdlib only); optional: anthropic, openai
"""

from __future__ import annotations

import re
import textwrap
import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class ExampleKind(str, Enum):
    """Distinguish positive (show this) from negative (avoid this) examples."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class Example:
    """
    A single demonstration for few-shot or negative-example prompting.

    Attributes
    ----------
    input_text  : The user input shown to the model.
    output_text : The ideal (positive) or bad (negative) response.
    kind        : POSITIVE = do this; NEGATIVE = do NOT do this.
    explanation : Optional coaching note added after negative examples.
                  Tells the model *why* the negative output is wrong.
    label       : Short display label (e.g. "Example 1", "Bad example").
    """
    input_text:  str
    output_text: str
    kind:        ExampleKind = ExampleKind.POSITIVE
    explanation: str = ""
    label:       str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = ("Example" if self.kind == ExampleKind.POSITIVE
                          else "Counter-example")


@dataclass
class ChainOfThoughtConfig:
    """
    Controls how Chain-of-Thought reasoning is injected into the prompt.

    Attributes
    ----------
    enabled        : Toggle CoT on/off without removing the config.
    steps          : Explicit reasoning steps to scaffold
                     (empty = free-form "think step by step").
    scratchpad_tag : XML tag wrapping the model's reasoning output.
    answer_tag     : XML tag wrapping the final answer.
    show_steps_in_prompt : Whether to print the steps list in the prompt
                           (True) or only instruct the model to reason (False).
    """
    enabled:              bool       = True
    steps:                list[str]  = field(default_factory=list)
    scratchpad_tag:       str        = "reasoning"
    answer_tag:           str        = "answer"
    show_steps_in_prompt: bool       = True


@dataclass
class XMLSchema:
    """
    Describes the XML structure expected in the model's output.

    Attributes
    ----------
    root_tag   : Outermost XML tag (e.g. "response").
    fields     : Ordered list of (tag, description) pairs.
    required   : Tags the model must always populate.
    attributes : Optional XML attributes per tag {tag: {attr: description}}.
    """
    root_tag:   str
    fields:     list[tuple[str, str]]          = field(default_factory=list)
    required:   list[str]                      = field(default_factory=list)
    attributes: dict[str, dict[str, str]]      = field(default_factory=dict)

    def render(self) -> str:
        """Return a human-readable schema description for injection into prompts."""
        lines = [f"Your response MUST be valid XML wrapped in <{self.root_tag}>...</{self.root_tag}>."]
        lines.append("Include these child elements (in order):")
        for tag, desc in self.fields:
            req = " [REQUIRED]" if tag in self.required else " [optional]"
            attrs = self.attributes.get(tag, {})
            attr_str = ""
            if attrs:
                attr_parts = [f'{k}="{v}"' for k, v in attrs.items()]
                attr_str = " — attributes: " + ", ".join(attr_parts)
            lines.append(f"  <{tag}>{req} — {desc}{attr_str}")
        lines.append("Output ONLY the XML block. Do not include markdown fences or prose.")
        return "\n".join(lines)

    def render_empty_template(self) -> str:
        """Return a blank XML template the model can fill in."""
        lines = [f"<{self.root_tag}>"]
        for tag, _ in self.fields:
            attrs = self.attributes.get(tag, {})
            attr_str = ""
            if attrs:
                attr_parts = [f'{k}="..."' for k in attrs]
                attr_str = " " + " ".join(attr_parts)
            lines.append(f"  <{tag}{attr_str}></{ tag}>")
        lines.append(f"</{self.root_tag}>")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class PromptTemplate:
    """
    A composable, reusable prompt template with advanced prompting techniques.

    Quick-start
    -----------
    >>> tmpl = PromptTemplate(name="sentiment")
    >>> tmpl.set_system("You are a sentiment analyst. Output XML only.")
    >>> tmpl.set_task("Classify the sentiment of the following review.")
    >>> tmpl.add_example(Example("Great product!", "<sentiment>Positive</sentiment>"))
    >>> tmpl.add_example(Example(
    ...     "ok i guess",
    ...     "<sentiment>Negative</sentiment>",
    ...     kind=ExampleKind.NEGATIVE,
    ...     explanation="'ok i guess' is ambiguous-neutral, not negative."
    ... ))
    >>> print(tmpl.render(review="Loved it, arrived fast!"))

    Template variables
    ------------------
    Use {variable_name} placeholders in task, system, or context strings.
    Pass keyword arguments to render() to fill them.

    >>> tmpl.set_task("Classify: {review}")
    >>> tmpl.render(review="Great product!")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str = "template",
        version: str = "1.0",
    ) -> None:
        self.name    = name
        self.version = version

        # Core prompt sections
        self._system:   str = ""
        self._task:     str = ""
        self._context:  str = ""
        self._constraints: list[str] = []

        # Examples (positive + negative, in insertion order)
        self._examples: list[Example] = []

        # Advanced technique configs
        self._cot:        Optional[ChainOfThoughtConfig] = None
        self._xml_schema: Optional[XMLSchema]            = None

        # Rendering options
        self._xml_wrap_input:  bool = False   # wrap user input in <input> tags
        self._input_tag:       str  = "input"
        self._role:            str  = ""      # role prompting string

        # Metadata / registry support
        self._tags: list[str] = []

    # ------------------------------------------------------------------
    # Fluent setters (return self for chaining)
    # ------------------------------------------------------------------

    def set_system(self, system_prompt: str) -> "PromptTemplate":
        """Set the system message. Supports {variable} interpolation."""
        self._system = textwrap.dedent(system_prompt).strip()
        return self

    def set_task(self, task: str) -> "PromptTemplate":
        """Set the core task instruction. Supports {variable} interpolation."""
        self._task = textwrap.dedent(task).strip()
        return self

    def set_context(self, context: str) -> "PromptTemplate":
        """Add background context injected before examples."""
        self._context = textwrap.dedent(context).strip()
        return self

    def set_role(self, role: str) -> "PromptTemplate":
        """
        Set a role persona string prepended to the system message.

        Example
        -------
        >>> tmpl.set_role("You are Dr. Lin, a senior NLP researcher with 12 years "
        ...               "of experience in sentiment analysis for e-commerce.")
        """
        self._role = textwrap.dedent(role).strip()
        return self

    def add_constraint(self, constraint: str) -> "PromptTemplate":
        """Add a numbered constraint/rule appended to the system message."""
        self._constraints.append(constraint.strip())
        return self

    def add_example(self, example: Example) -> "PromptTemplate":
        """Append a positive or negative demonstration example."""
        self._examples.append(deepcopy(example))
        return self

    def set_cot(self, config: Optional[ChainOfThoughtConfig] = None) -> "PromptTemplate":
        """
        Enable Chain-of-Thought reasoning.

        Pass a ChainOfThoughtConfig for fine-grained control, or None to use
        sensible defaults (free-form "think step by step" with XML scratchpad).
        """
        self._cot = config if config is not None else ChainOfThoughtConfig()
        return self

    def set_xml_schema(self, schema: XMLSchema) -> "PromptTemplate":
        """Define the expected XML output structure."""
        self._xml_schema = schema
        return self

    def wrap_input_xml(self, enabled: bool = True, tag: str = "input") -> "PromptTemplate":
        """Wrap the user's input text in an XML tag for unambiguous parsing."""
        self._xml_wrap_input = enabled
        self._input_tag = tag
        return self

    def add_tag(self, *tags: str) -> "PromptTemplate":
        """Tag the template for registry/search purposes."""
        self._tags.extend(tags)
        return self

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **variables: Any) -> dict[str, str]:
        """
        Render the complete prompt, returning a dict with 'system' and 'user' keys.

        Parameters
        ----------
        **variables : Values for {placeholder} tokens in the template strings.

        Returns
        -------
        dict with keys:
            "system" — the system message (empty string if none set)
            "user"   — the full user-turn message

        Raises
        ------
        PromptRenderError  if required placeholders are missing.

        Example
        -------
        >>> msgs = tmpl.render(review="Loved it!")
        >>> # Use with any LLM client:
        >>> client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     system=msgs["system"],
        ...     messages=[{"role": "user", "content": msgs["user"]}]
        ... )
        """
        try:
            system = self._render_system(variables)
            user   = self._render_user(variables)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise PromptRenderError(
                f"Template '{self.name}' is missing variable: '{missing}'. "
                f"Pass it as a keyword argument to render()."
            ) from exc

        return {"system": system, "user": user}

    def render_string(self, **variables: Any) -> str:
        """
        Render as a single concatenated string (system + user).
        Useful for models / APIs that don't support separate roles.
        """
        parts = self.render(**variables)
        sections = []
        if parts["system"]:
            sections.append(f"[SYSTEM]\n{parts['system']}")
        if parts["user"]:
            sections.append(f"[USER]\n{parts['user']}")
        return "\n\n".join(sections)

    def render_messages(self, **variables: Any) -> list[dict[str, str]]:
        """
        Render as an OpenAI / Anthropic-style messages list.

        Returns
        -------
        list of {"role": ..., "content": ...} dicts.
        If no system message is set, returns only the user message.
        """
        parts = self.render(**variables)
        msgs = []
        if parts["system"]:
            msgs.append({"role": "system", "content": parts["system"]})
        msgs.append({"role": "user", "content": parts["user"]})
        return msgs

    # ------------------------------------------------------------------
    # Private render helpers
    # ------------------------------------------------------------------

    def _render_system(self, variables: dict[str, Any]) -> str:
        parts: list[str] = []

        if self._role:
            parts.append(self._role.format(**variables))

        if self._system:
            parts.append(self._system.format(**variables))

        if self._constraints:
            parts.append("Rules you MUST follow:")
            for i, c in enumerate(self._constraints, 1):
                parts.append(f"  {i}. {c.format(**variables)}")

        if self._xml_schema:
            parts.append("")
            parts.append(self._xml_schema.render())
            parts.append("")
            parts.append("Empty template for reference:")
            parts.append(self._xml_schema.render_empty_template())

        return "\n\n".join(p for p in parts if p.strip())

    def _render_user(self, variables: dict[str, Any]) -> str:
        parts: list[str] = []

        if self._context:
            parts.append(self._context.format(**variables))

        # ── CoT step list (if show_steps_in_prompt) ──────────────────
        if (self._cot and self._cot.enabled
                and self._cot.steps and self._cot.show_steps_in_prompt):
            parts.append("To answer correctly, follow these reasoning steps:")
            for i, step in enumerate(self._cot.steps, 1):
                parts.append(f"  {i}. {step}")

        # ── Examples ─────────────────────────────────────────────────
        pos_examples = [e for e in self._examples if e.kind == ExampleKind.POSITIVE]
        neg_examples = [e for e in self._examples if e.kind == ExampleKind.NEGATIVE]

        if pos_examples:
            parts.append("--- Examples ---")
            for ex in pos_examples:
                parts.append(self._render_example(ex, variables))

        if neg_examples:
            parts.append("--- Counter-examples (do NOT produce outputs like these) ---")
            for ex in neg_examples:
                parts.append(self._render_example(ex, variables))

        # ── Task instruction ─────────────────────────────────────────
        if self._task:
            parts.append("--- Task ---")
            parts.append(self._task.format(**variables))

        # ── CoT invocation ───────────────────────────────────────────
        if self._cot and self._cot.enabled:
            parts.append(self._render_cot_invocation())

        return "\n\n".join(p for p in parts if p.strip())

    def _render_example(self, ex: Example, variables: dict[str, Any]) -> str:
        inp = ex.input_text.format(**variables) if "{" in ex.input_text else ex.input_text
        out = ex.output_text.format(**variables) if "{" in ex.output_text else ex.output_text

        if self._xml_wrap_input:
            inp_display = f"<{self._input_tag}>{inp}</{self._input_tag}>"
        else:
            inp_display = inp

        lines = [f"[{ex.label}]", f"Input:  {inp_display}", f"Output: {out}"]

        if ex.kind == ExampleKind.NEGATIVE and ex.explanation:
            lines.append(f"Why wrong: {ex.explanation}")

        return "\n".join(lines)

    def _render_cot_invocation(self) -> str:
        tag_r = self._cot.scratchpad_tag
        tag_a = self._cot.answer_tag

        if self._cot.steps and not self._cot.show_steps_in_prompt:
            # Steps guide reasoning but aren't shown in the prompt body
            step_block = "\n".join(f"  {i}. {s}" for i, s in enumerate(self._cot.steps, 1))
            reasoning_instruction = (
                f"Think through the problem using these steps:\n{step_block}"
            )
        elif self._cot.steps:
            reasoning_instruction = "Apply the reasoning steps listed above."
        else:
            reasoning_instruction = "Think step by step before answering."

        return (
            f"{reasoning_instruction}\n"
            f"Show your work inside <{tag_r}>...</{tag_r}> tags, "
            f"then output your final answer inside <{tag_a}>...</{tag_a}> tags."
        )

    # ------------------------------------------------------------------
    # Introspection & utilities
    # ------------------------------------------------------------------

    def get_variables(self) -> set[str]:
        """Return the set of {placeholder} names found across all template strings."""
        all_text = " ".join([
            self._system, self._task, self._context, self._role,
            *self._constraints,
            *[e.input_text + e.output_text for e in self._examples],
        ])
        return set(re.findall(r"\{(\w+)\}", all_text))

    def validate(self, **variables: Any) -> list[str]:
        """
        Dry-run validation. Returns a list of warning strings (empty = OK).

        Checks performed
        ----------------
        • All {placeholder} variables are supplied
        • At least one positive example if few-shot is implied
        • Negative examples have explanations
        • CoT tags don't clash with XML schema tags
        """
        warnings: list[str] = []
        supplied = set(variables.keys())
        required = self.get_variables()
        missing  = required - supplied
        if missing:
            warnings.append(f"Missing variables: {missing}")

        if self._examples:
            neg_no_exp = [
                e.label for e in self._examples
                if e.kind == ExampleKind.NEGATIVE and not e.explanation
            ]
            if neg_no_exp:
                warnings.append(
                    f"Negative example(s) missing explanation: {neg_no_exp}. "
                    "Add an explanation= to help the model understand why they're wrong."
                )

        if self._cot and self._xml_schema:
            cot_tags  = {self._cot.scratchpad_tag, self._cot.answer_tag}
            xml_tags  = {t for t, _ in self._xml_schema.fields}
            collisions = cot_tags & xml_tags
            if collisions:
                warnings.append(
                    f"CoT tag(s) {collisions} collide with XML schema field tags. "
                    "Rename them to avoid parser confusion."
                )

        return warnings

    def summary(self) -> str:
        """Return a human-readable summary of the template configuration."""
        pos = sum(1 for e in self._examples if e.kind == ExampleKind.POSITIVE)
        neg = sum(1 for e in self._examples if e.kind == ExampleKind.NEGATIVE)
        lines = [
            f"PromptTemplate  '{self.name}'  v{self.version}",
            f"  Role:         {'set' if self._role else 'not set'}",
            f"  System msg:   {'set' if self._system else 'not set'}",
            f"  Task:         {'set' if self._task else 'not set'}",
            f"  Context:      {'set' if self._context else 'not set'}",
            f"  Constraints:  {len(self._constraints)}",
            f"  Examples:     {pos} positive, {neg} negative",
            f"  CoT:          {'enabled' if self._cot and self._cot.enabled else 'disabled'}",
            f"  XML schema:   {'set (' + self._xml_schema.root_tag + ')' if self._xml_schema else 'not set'}",
            f"  Variables:    {sorted(self.get_variables()) or 'none'}",
            f"  Tags:         {self._tags or 'none'}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the template to a plain dict (JSON-safe)."""
        return {
            "name":    self.name,
            "version": self.version,
            "system":  self._system,
            "role":    self._role,
            "task":    self._task,
            "context": self._context,
            "constraints": self._constraints,
            "tags":    self._tags,
            "examples": [
                {
                    "input_text":  e.input_text,
                    "output_text": e.output_text,
                    "kind":        e.kind.value,
                    "explanation": e.explanation,
                    "label":       e.label,
                }
                for e in self._examples
            ],
            "cot": (
                {
                    "enabled":              self._cot.enabled,
                    "steps":                self._cot.steps,
                    "scratchpad_tag":       self._cot.scratchpad_tag,
                    "answer_tag":           self._cot.answer_tag,
                    "show_steps_in_prompt": self._cot.show_steps_in_prompt,
                }
                if self._cot else None
            ),
            "xml_schema": (
                {
                    "root_tag":   self._xml_schema.root_tag,
                    "fields":     self._xml_schema.fields,
                    "required":   self._xml_schema.required,
                    "attributes": self._xml_schema.attributes,
                }
                if self._xml_schema else None
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptTemplate":
        """Deserialise from a plain dict produced by to_dict()."""
        tmpl = cls(name=data.get("name", "template"),
                   version=data.get("version", "1.0"))
        tmpl._system      = data.get("system", "")
        tmpl._role        = data.get("role", "")
        tmpl._task        = data.get("task", "")
        tmpl._context     = data.get("context", "")
        tmpl._constraints = data.get("constraints", [])
        tmpl._tags        = data.get("tags", [])

        for ex in data.get("examples", []):
            tmpl.add_example(Example(
                input_text  = ex["input_text"],
                output_text = ex["output_text"],
                kind        = ExampleKind(ex.get("kind", "positive")),
                explanation = ex.get("explanation", ""),
                label       = ex.get("label", ""),
            ))

        if cot := data.get("cot"):
            tmpl._cot = ChainOfThoughtConfig(
                enabled              = cot.get("enabled", True),
                steps                = cot.get("steps", []),
                scratchpad_tag       = cot.get("scratchpad_tag", "reasoning"),
                answer_tag           = cot.get("answer_tag", "answer"),
                show_steps_in_prompt = cot.get("show_steps_in_prompt", True),
            )

        if xs := data.get("xml_schema"):
            tmpl._xml_schema = XMLSchema(
                root_tag   = xs["root_tag"],
                fields     = [tuple(f) for f in xs.get("fields", [])],
                required   = xs.get("required", []),
                attributes = xs.get("attributes", {}),
            )

        return tmpl

    @classmethod
    def from_json(cls, json_str: str) -> "PromptTemplate":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Copy & composition
    # ------------------------------------------------------------------

    def copy(self, new_name: Optional[str] = None) -> "PromptTemplate":
        """Return a deep copy, optionally with a new name."""
        cloned = PromptTemplate.from_dict(self.to_dict())
        if new_name:
            cloned.name = new_name
        return cloned

    def merge(self, other: "PromptTemplate") -> "PromptTemplate":
        """
        Return a new template that merges self with other.
        other's settings override self's where both define the same field.
        Examples from both are combined (self first, then other).
        """
        base = self.copy(new_name=f"{self.name}+{other.name}")

        if other._role:        base._role    = other._role
        if other._system:      base._system  = other._system
        if other._task:        base._task    = other._task
        if other._context:     base._context = other._context
        if other._cot:         base._cot     = deepcopy(other._cot)
        if other._xml_schema:  base._xml_schema = deepcopy(other._xml_schema)

        base._constraints.extend(other._constraints)
        base._examples.extend(deepcopy(e) for e in other._examples)
        base._tags = list(set(base._tags + other._tags))

        return base

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(name={self.name!r}, "
            f"examples={len(self._examples)}, "
            f"cot={'on' if self._cot and self._cot.enabled else 'off'}, "
            f"xml={'on' if self._xml_schema else 'off'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error class
# ─────────────────────────────────────────────────────────────────────────────

class PromptRenderError(ValueError):
    """Raised when render() cannot fill all template variables."""


# ─────────────────────────────────────────────────────────────────────────────
# Template registry
# ─────────────────────────────────────────────────────────────────────────────

class TemplateRegistry:
    """
    A simple in-memory registry for managing multiple PromptTemplate instances.

    Usage
    -----
    >>> registry = TemplateRegistry()
    >>> registry.register(my_template)
    >>> t = registry.get("sentiment-v2")
    >>> registry.list_templates()
    """

    def __init__(self) -> None:
        self._store: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate, overwrite: bool = False) -> None:
        key = f"{template.name}-v{template.version}"
        if key in self._store and not overwrite:
            raise KeyError(f"Template '{key}' already registered. Use overwrite=True.")
        self._store[key] = template

    def get(self, name: str, version: str = "1.0") -> PromptTemplate:
        key = f"{name}-v{version}"
        if key not in self._store:
            raise KeyError(f"Template '{key}' not found in registry.")
        return self._store[key]

    def list_templates(self) -> list[str]:
        return sorted(self._store.keys())

    def search_by_tag(self, tag: str) -> list[PromptTemplate]:
        return [t for t in self._store.values() if tag in t._tags]

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"TemplateRegistry({len(self)} templates: {self.list_templates()})"


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built template factories
# ─────────────────────────────────────────────────────────────────────────────

def make_sentiment_template() -> PromptTemplate:
    """
    Pre-built template: sentiment analysis with XML output + negative examples.

    Demonstrates:  few-shot, negative examples, XML schema, role prompting.
    """
    schema = XMLSchema(
        root_tag = "review_analysis",
        fields   = [
            ("sentiment",   "One of: Positive | Neutral | Negative"),
            ("confidence",  "One of: High | Medium | Low"),
            ("key_phrases", "Comma-separated list of the 1–3 most influential phrases"),
            ("reasoning",   "One sentence explaining the classification"),
        ],
        required = ["sentiment", "confidence"],
    )

    tmpl = (
        PromptTemplate(name="sentiment", version="2.0")
        .set_role(
            "You are a senior product analytics engineer specialising in "
            "e-commerce review sentiment. Your classifications feed directly "
            "into a customer support triage pipeline."
        )
        .set_system(
            "Analyse the sentiment of customer reviews.\n"
            "Be precise: 'Neutral' is for genuinely mixed or ambiguous reviews, "
            "not a default when uncertain."
        )
        .set_xml_schema(schema)
        .add_constraint("Output ONLY the XML block — no markdown, no prose.")
        .add_constraint("Never infer intent beyond what is explicitly stated in the review.")
        .add_example(Example(
            input_text  = "The noise-cancelling is incredible and battery lasts forever.",
            output_text = (
                "<review_analysis>\n"
                "  <sentiment>Positive</sentiment>\n"
                "  <confidence>High</confidence>\n"
                "  <key_phrases>noise-cancelling is incredible, battery lasts forever</key_phrases>\n"
                "  <reasoning>Both mentioned features are explicitly praised with strong language.</reasoning>\n"
                "</review_analysis>"
            ),
            label = "Positive example",
        ))
        .add_example(Example(
            input_text  = "Sounds good but broke after 2 months.",
            output_text = (
                "<review_analysis>\n"
                "  <sentiment>Negative</sentiment>\n"
                "  <confidence>High</confidence>\n"
                "  <key_phrases>broke after 2 months</key_phrases>\n"
                "  <reasoning>Despite audio praise, product failure dominates the sentiment.</reasoning>\n"
                "</review_analysis>"
            ),
            label = "Mixed-leaning-negative example",
        ))
        .add_example(Example(
            input_text  = "Sounds good but broke after 2 months.",
            output_text = (
                "<review_analysis>\n"
                "  <sentiment>Neutral</sentiment>\n"
                "  <confidence>High</confidence>\n"
                "</review_analysis>"
            ),
            kind        = ExampleKind.NEGATIVE,
            explanation = (
                "Labelling product failure as Neutral underweights the severity. "
                "Hardware failure after 2 months is a strong negative signal."
            ),
            label = "Counter-example: do not label product failure as Neutral",
        ))
        .set_task("Analyse this review:\n\n{review}")
        .wrap_input_xml(enabled=False)
        .add_tag("sentiment", "xml", "few-shot", "negative-examples", "ecommerce")
    )
    return tmpl


def make_cot_math_template() -> PromptTemplate:
    """
    Pre-built template: multi-step math reasoning with Chain-of-Thought.

    Demonstrates:  CoT with explicit steps, XML answer tag, variable input.
    """
    cot = ChainOfThoughtConfig(
        enabled              = True,
        steps                = [
            "Re-read the problem and identify all given values and the unknown.",
            "Determine which formula or approach applies.",
            "Execute each calculation, showing intermediate results.",
            "Sanity-check the answer against the problem constraints.",
        ],
        scratchpad_tag       = "reasoning",
        answer_tag           = "answer",
        show_steps_in_prompt = True,
    )

    return (
        PromptTemplate(name="cot-math", version="1.0")
        .set_system(
            "You are a patient math tutor. Walk students through every step "
            "of a solution so they can learn the reasoning, not just the answer."
        )
        .set_cot(cot)
        .set_task("{problem}")
        .add_tag("math", "cot", "education")
    )


def make_code_review_template() -> PromptTemplate:
    """
    Pre-built template: structured code review with XML output + CoT.

    Demonstrates:  role + system + CoT + XML schema combined.
    """
    schema = XMLSchema(
        root_tag   = "code_review",
        fields     = [
            ("summary",        "2-sentence overall assessment"),
            ("severity",       "One of: Critical | Major | Minor | Clean"),
            ("issues",         "List of specific issues found"),
            ("suggestions",    "Concrete improvement recommendations"),
            ("positive_notes", "What the code does well"),
        ],
        required   = ["summary", "severity"],
        attributes = {"issues": {"count": "number of issues found"}},
    )

    cot = ChainOfThoughtConfig(
        enabled              = True,
        steps                = [
            "Identify correctness issues (bugs, logic errors, off-by-ones).",
            "Assess security concerns (injections, unvalidated input, secrets in code).",
            "Evaluate readability and naming conventions.",
            "Check for performance anti-patterns.",
            "Note strengths before summarising.",
        ],
        scratchpad_tag       = "analysis_notes",
        answer_tag           = "code_review",
        show_steps_in_prompt = True,
    )

    return (
        PromptTemplate(name="code-review", version="1.0")
        .set_role(
            "You are a principal software engineer conducting thorough, "
            "constructive code reviews. You are direct but always respectful."
        )
        .set_system("Review the provided code and produce a structured assessment.")
        .set_cot(cot)
        .set_xml_schema(schema)
        .add_constraint("Never invent issues that aren't clearly present in the code.")
        .add_constraint("If the code is clean, say so explicitly — don't manufacture nitpicks.")
        .set_task(
            "Language: {language}\n\n"
            "```{language}\n{code}\n```"
        )
        .add_tag("code-review", "engineering", "xml", "cot")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo / usage examples
# ─────────────────────────────────────────────────────────────────────────────

def _demo_sentiment() -> None:
    print("=" * 70)
    print("DEMO 1: Sentiment Template (XML + few-shot + negative examples)")
    print("=" * 70)

    tmpl = make_sentiment_template()
    print(tmpl.summary())
    print()

    warnings = tmpl.validate(review="The sound quality is decent but the build feels cheap.")
    if warnings:
        print("Validation warnings:", warnings)
    else:
        print("Validation: OK ✓")
    print()

    rendered = tmpl.render_string(review="The sound quality is decent but the build feels cheap.")
    print(rendered)


def _demo_cot() -> None:
    print("\n" + "=" * 70)
    print("DEMO 2: Chain-of-Thought Math Template")
    print("=" * 70)

    tmpl = make_cot_math_template()
    rendered = tmpl.render_string(
        problem="A train travels 240 km at 80 km/h, then 180 km at 60 km/h. "
                "What is its average speed for the whole journey?"
    )
    print(rendered)


def _demo_code_review() -> None:
    print("\n" + "=" * 70)
    print("DEMO 3: Code Review Template (Role + CoT + XML)")
    print("=" * 70)

    tmpl = make_code_review_template()
    code_snippet = (
        "def get_user(user_id):\n"
        "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "    return db.execute(query)"
    )
    rendered = tmpl.render_string(language="python", code=code_snippet)
    print(rendered)


def _demo_serialisation() -> None:
    print("\n" + "=" * 70)
    print("DEMO 4: Serialisation round-trip (to_json / from_json)")
    print("=" * 70)

    original = make_sentiment_template()
    json_str  = original.to_json()
    restored  = PromptTemplate.from_json(json_str)

    orig_out = original.render(review="Great product!")
    rest_out = restored.render(review="Great product!")
    match    = orig_out == rest_out

    print(f"Round-trip match: {match} ✓" if match else f"Round-trip MISMATCH ✗")
    print(f"JSON size: {len(json_str)} bytes")


def _demo_registry() -> None:
    print("\n" + "=" * 70)
    print("DEMO 5: TemplateRegistry")
    print("=" * 70)

    registry = TemplateRegistry()
    registry.register(make_sentiment_template())
    registry.register(make_cot_math_template())
    registry.register(make_code_review_template())

    print(registry)
    print("Templates tagged 'xml':", [t.name for t in registry.search_by_tag("xml")])
    print("Templates tagged 'cot':", [t.name for t in registry.search_by_tag("cot")])


if __name__ == "__main__":
    _demo_sentiment()
    _demo_cot()
    _demo_code_review()
    _demo_serialisation()
    _demo_registry()
