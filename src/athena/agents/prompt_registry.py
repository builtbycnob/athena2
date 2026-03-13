# src/athena/agents/prompt_registry.py
"""Prompt template registry with composable templates and override support.

Provides a generic `build_party_prompt()` that replaces per-agent prompt builders.
Current Italian prompts are registered as overrides for zero quality regression.
"""

import json
from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    role_type: str           # "advocate" | "adjudicator"
    system_template: str     # with {placeholders}
    output_format: str       # JSON structure description (embedded in system_template)
    constraints: str         # role-specific constraints (embedded in system_template)
    user_preamble: str = ""  # opening line of user prompt
    user_closing: str = ""   # closing line of user prompt
    context_blocks: list[str] = field(default_factory=list)  # ordered context block labels


_PROMPT_REGISTRY: dict[str, PromptTemplate] = {}


def register_prompt(key: str, template: PromptTemplate) -> None:
    """Register a prompt template or override."""
    _PROMPT_REGISTRY[key] = template


def get_prompt(key: str) -> PromptTemplate:
    """Retrieve a registered prompt template."""
    if key not in _PROMPT_REGISTRY:
        raise KeyError(f"Prompt template '{key}' not found in registry. "
                       f"Available: {list(_PROMPT_REGISTRY.keys())}")
    return _PROMPT_REGISTRY[key]


def list_prompts() -> list[str]:
    """List all registered prompt keys."""
    return list(_PROMPT_REGISTRY.keys())


def _format_context_block(label: str, data) -> str:
    """Format a context block for injection into prompts."""
    return f"\n## {label}\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```\n"


def build_party_prompt(
    context: dict,
    prompt_key: str,
    template_vars: dict | None = None,
) -> tuple[str, str]:
    """Build system + user prompt from registry template.

    Args:
        context: Agent context dict from build_party_context/build_adjudicator_context.
        prompt_key: Registry lookup key.
        template_vars: Variables to substitute in system_template (e.g. {advocacy_style}).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    template = get_prompt(prompt_key)
    tvars = template_vars or {}

    # Build system prompt by substituting placeholders
    system = template.system_template
    for k, v in tvars.items():
        system = system.replace(f"{{{k}}}", str(v))

    # Build user prompt from context blocks
    user_parts = [template.user_preamble] if template.user_preamble else []
    for label in template.context_blocks:
        # Map label to context key
        key = _LABEL_TO_KEY.get(label, label.lower().replace(" ", "_"))
        if key in context:
            user_parts.append(_format_context_block(label, context[key]))
    user_parts.append(template.user_closing)

    return system, "\n".join(user_parts)


# Mapping from Italian labels to context dict keys
_LABEL_TO_KEY = {
    "Fatti": "facts",
    "Prove": "evidence",
    "Testi normativi": "legal_texts",
    "Precedenti": "precedents",
    "Seed arguments": "seed_arguments",
    "Seed arguments difensivi": "seed_arguments",
    "Obiettivi della tua parte": "own_party",
    "Stakes": "stakes",
    "Regole procedurali": "procedural_rules",
    "Memoria dell'opponente (depositata)": "appellant_brief",
    "Memoria del Comune (depositata)": "respondent_brief",
    # Swiss labels
    "Memoria del ricorrente (depositata)": "appellant_brief",
    "Memoria della controparte (depositata)": "respondent_brief",
    "Insight dal Knowledge Graph": "kg_insights",
    # Two-step judge
    "Errori identificati (fase precedente)": "step1_errors",
    # RAG
    "Testi normativi aggiuntivi (RAG)": "rag_legal_texts",
}
