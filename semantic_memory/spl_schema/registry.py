"""
SPL Schema Registry — V2

A schema defines which slot types are valid for a domain, how many items each
slot can hold, what aliases are accepted in the input, and what domain tags apply.

Design constraint: schemas EXTEND core SPL, they don't replace it.
Core slots (task, pref, !pref, fact, ent, q_hist) are always present.
Domain extensions add or restrict slots without breaking the core format.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlotDef:
    """Definition for one SPL slot type within a schema."""
    name: str                           # canonical name (e.g. "task")
    max_items: int = 4                  # max items rendered in SPL output
    spl_prefix: str = ""                # prefix in the encoded string (e.g. "task=")
    aliases: tuple[str, ...] = ()       # alternative input names accepted
    domain_tags: tuple[str, ...] = ()   # tags that describe which domains use this slot
    description: str = ""


@dataclass
class SPLSchema:
    """A versioned SPL schema for a specific domain."""
    name: str
    version: str
    description: str
    slots: list[SlotDef]
    extends: str = "general"           # parent schema name ("general" for built-ins)

    def get_slot(self, name: str) -> SlotDef | None:
        for slot in self.slots:
            if slot.name == name or name in slot.aliases:
                return slot
        return None

    def slot_names(self) -> set[str]:
        return {s.name for s in self.slots}

    def max_items_for(self, slot_name: str) -> int:
        slot = self.get_slot(slot_name)
        return slot.max_items if slot else 0


class SPLSchemaRegistry:
    """Singleton registry of named SPL schemas."""

    _schemas: dict[str, SPLSchema] = {}

    @classmethod
    def register(cls, schema: SPLSchema) -> None:
        cls._schemas[schema.name] = schema

    @classmethod
    def get(cls, name: str) -> SPLSchema:
        if name not in cls._schemas:
            raise KeyError(f"SPL schema '{name}' not found. Available: {list(cls._schemas)}")
        return cls._schemas[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._schemas.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Built-in schemas
# ─────────────────────────────────────────────────────────────────────────────

_GENERAL = SPLSchema(
    name="general",
    version="2.0",
    description="Core SPL schema. Baseline for all domains.",
    extends="",
    slots=[
        SlotDef("task",       max_items=3, spl_prefix="task=",   domain_tags=("all",),       description="Primary tasks/goals"),
        SlotDef("constraint", max_items=2, spl_prefix="",        domain_tags=("all",),       description="Deadlines and hard constraints"),
        SlotDef("preference", max_items=4, spl_prefix="pref=",   domain_tags=("all",),       description="Positive preferences"),
        SlotDef("neg_pref",   max_items=4, spl_prefix="!pref=",  aliases=("negative_pref",), domain_tags=("all",), description="Negative preferences"),
        SlotDef("fact",       max_items=4, spl_prefix="",        domain_tags=("all",),       description="Known facts (hw, scope, status)"),
        SlotDef("entity",     max_items=4, spl_prefix="ent=",    domain_tags=("all",),       description="Named entities"),
        SlotDef("event",      max_items=4, spl_prefix="q_hist=", aliases=("topic",),         domain_tags=("all",), description="Query topic history"),
    ],
)

_CODING = SPLSchema(
    name="coding",
    version="2.0",
    description="Extended schema for software engineering conversations.",
    extends="general",
    slots=[
        SlotDef("task",       max_items=4, spl_prefix="task=",   domain_tags=("coding",),   description="Coding tasks"),
        SlotDef("constraint", max_items=3, spl_prefix="",        domain_tags=("coding",),   description="Deadlines, version pinning, CI constraints"),
        SlotDef("preference", max_items=5, spl_prefix="pref=",   domain_tags=("coding",),   description="Framework/tool preferences"),
        SlotDef("neg_pref",   max_items=5, spl_prefix="!pref=",  aliases=("negative_pref",), domain_tags=("coding",)),
        SlotDef("fact",       max_items=5, spl_prefix="",        domain_tags=("coding",),   description="hw, scope, language version, os"),
        SlotDef("entity",     max_items=5, spl_prefix="ent=",    domain_tags=("coding",),   description="Libraries, repos, services"),
        SlotDef("event",      max_items=5, spl_prefix="q_hist=", aliases=("topic",),        domain_tags=("coding",)),
        SlotDef("debug",      max_items=2, spl_prefix="err=",    domain_tags=("coding",),   description="Active error types / stack trace tokens"),
        SlotDef("stack",      max_items=3, spl_prefix="stack=",  domain_tags=("coding",),   description="Tech stack components"),
    ],
)

_MEDICAL = SPLSchema(
    name="medical",
    version="2.0",
    description="Schema for clinical/health-domain conversations.",
    extends="general",
    slots=[
        SlotDef("task",       max_items=2, spl_prefix="task=",    domain_tags=("medical",)),
        SlotDef("constraint", max_items=3, spl_prefix="",         domain_tags=("medical",), description="Contraindications, allergies"),
        SlotDef("preference", max_items=3, spl_prefix="pref=",    domain_tags=("medical",)),
        SlotDef("neg_pref",   max_items=3, spl_prefix="!pref=",   aliases=("negative_pref",), domain_tags=("medical",)),
        SlotDef("fact",       max_items=6, spl_prefix="",         domain_tags=("medical",), description="Patient facts: age, condition, dosage"),
        SlotDef("entity",     max_items=4, spl_prefix="ent=",     domain_tags=("medical",), description="Medications, conditions, procedures"),
        SlotDef("event",      max_items=3, spl_prefix="q_hist=",  domain_tags=("medical",)),
        SlotDef("symptom",    max_items=4, spl_prefix="sym=",     domain_tags=("medical",), description="Reported symptoms"),
        SlotDef("vitals",     max_items=3, spl_prefix="vitals=",  domain_tags=("medical",), description="Key vitals if provided"),
    ],
)

_LEGAL = SPLSchema(
    name="legal",
    version="2.0",
    description="Schema for legal research and document analysis conversations.",
    extends="general",
    slots=[
        SlotDef("task",       max_items=3, spl_prefix="task=",    domain_tags=("legal",)),
        SlotDef("constraint", max_items=4, spl_prefix="",         domain_tags=("legal",), description="Jurisdictions, filing deadlines"),
        SlotDef("preference", max_items=3, spl_prefix="pref=",    domain_tags=("legal",)),
        SlotDef("neg_pref",   max_items=3, spl_prefix="!pref=",   aliases=("negative_pref",), domain_tags=("legal",)),
        SlotDef("fact",       max_items=5, spl_prefix="",         domain_tags=("legal",), description="Case facts, parties, dates"),
        SlotDef("entity",     max_items=5, spl_prefix="ent=",     domain_tags=("legal",), description="Statutes, parties, courts"),
        SlotDef("event",      max_items=3, spl_prefix="q_hist=",  domain_tags=("legal",)),
        SlotDef("jurisdiction", max_items=2, spl_prefix="juris=", domain_tags=("legal",), description="Applicable jurisdiction(s)"),
        SlotDef("clause",     max_items=3, spl_prefix="clause=",  domain_tags=("legal",), description="Relevant contract/statute clauses"),
    ],
)

# Register all built-ins
for _schema in (_GENERAL, _CODING, _MEDICAL, _LEGAL):
    SPLSchemaRegistry.register(_schema)
