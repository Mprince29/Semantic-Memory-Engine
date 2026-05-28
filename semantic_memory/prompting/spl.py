from collections import defaultdict

from semantic_memory.domain.models import MEMORY_STATE_CONTEXTUAL, SemanticMemoryObject
from semantic_memory.spl_schema.registry import SPLSchema, SPLSchemaRegistry


class SPLEncoder:
    type_priority = {
        "task": 1,
        "constraint": 2,
        "preference": 3,
        "entity": 4,
        "event": 5,
        "fact": 6,
    }

    def encode(
        self,
        smos: list[SemanticMemoryObject],
        query: str,
        schema_name: str = "general",
        contextual: list[SemanticMemoryObject] | None = None,
    ) -> str:
        """
        Encode memories as SPL using the named schema.

        Active memories fill core + extension slots.
        Contextual memories (scope-disputed) appear in a separate ?pref= slot
        so the model knows the signal exists but is context-dependent.
        """
        schema = SPLSchemaRegistry.get(schema_name)
        grouped: dict[str, list[SemanticMemoryObject]] = defaultdict(list)
        # Use the contextual list if provided; otherwise extract from smos by state
        self._contextual = contextual or [
            s for s in smos if s.memory_state == MEMORY_STATE_CONTEXTUAL
        ]
        active_smos = [s for s in smos if s.memory_state != MEMORY_STATE_CONTEXTUAL]
        for item in sorted(active_smos, key=lambda value: self.type_priority.get(value.type, 99)):
            grouped[item.type].append(item)

        lines: list[str] = []
        lines.extend(self._build_task_line(grouped, schema))
        lines.extend(self._build_preference_lines(grouped, schema))
        lines.extend(self._build_contextual_preference_line(self._contextual))
        lines.extend(self._build_fact_lines(grouped, schema))
        lines.extend(self._build_entity_lines(grouped, schema))
        lines.extend(self._build_history_lines(grouped, schema))
        lines.extend(self._build_extension_lines(grouped, schema))

        context = "\n".join(line for line in lines if line)
        return f"[CTX]\n{context}\n[/CTX]\n[Q] {query.strip()}"

    @staticmethod
    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75)

    def _build_task_line(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        limit = schema.max_items_for("task") or 3
        task_parts: list[str] = []
        for item in grouped.get("task", [])[:limit]:
            task_parts.append(f"task={self._normalize(item.value)}")
        constraint_limit = schema.max_items_for("constraint") or 2
        for item in grouped.get("constraint", [])[:constraint_limit]:
            if item.predicate == "deadline":
                task_parts.append(f"deadline={self._normalize(item.value)}")
        return [" ".join(task_parts)] if task_parts else []

    def _build_preference_lines(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        pref_limit = schema.max_items_for("preference") or 4
        pos = [self._normalize(item.value) for item in grouped.get("preference", []) if "pos" in item.predicate][:pref_limit]
        neg = [self._normalize(item.value) for item in grouped.get("preference", []) if "neg" in item.predicate][:pref_limit]
        lines: list[str] = []
        if pos:
            lines.append(f"pref=[{','.join(pos)}]")
        if neg:
            lines.append(f"!pref=[{','.join(neg)}]")
        return lines

    def _build_fact_lines(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        limit = schema.max_items_for("fact") or 4
        facts: list[str] = []
        seen_keys: set[str] = set()
        for item in grouped.get("fact", [])[:limit]:
            key = self._normalize(item.predicate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            facts.append(f"{key}={self._normalize(item.value)}")
        return [" ".join(facts)] if facts else []

    def _build_entity_lines(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        limit = schema.max_items_for("entity") or 4
        entities = [
            f"{self._normalize(item.subject)}:{self._normalize(item.value)}"
            for item in grouped.get("entity", [])[:limit]
        ]
        return [f"ent=[{','.join(entities)}]"] if entities else []

    def _build_history_lines(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        limit = schema.max_items_for("event") or 4
        history: list[str] = []
        seen: set[str] = set()
        for item in grouped.get("event", []):
            token = self._normalize(item.value)
            if token in seen:
                continue
            seen.add(token)
            history.append(token)
            if len(history) == limit:
                break
        return [f"q_hist=[{','.join(history)}]"] if history else []

    def _build_contextual_preference_line(
        self, contextual: list[SemanticMemoryObject]
    ) -> list[str]:
        """
        Emit ?pref=[...] for contextual memories so the model knows these
        signals exist but are scope-dependent (not universal constraints).
        """
        values = [
            self._normalize(item.value)
            for item in contextual
            if item.type == "preference"
        ][:4]
        return [f"?pref=[{','.join(values)}]"] if values else []

    def _build_extension_lines(
        self, grouped: dict[str, list[SemanticMemoryObject]], schema: SPLSchema
    ) -> list[str]:
        """Emit domain-specific extension slots not in core SPL."""
        core_types = {"task", "constraint", "preference", "entity", "event", "fact"}
        lines: list[str] = []
        for slot in schema.slots:
            if slot.name in core_types or slot.name == "neg_pref":
                continue
            # Extension slot items: match on type OR on predicate matching the slot name
            items = [
                s for s in grouped.get(slot.name, [])
                if s.memory_state != MEMORY_STATE_CONTEXTUAL
            ][:slot.max_items]
            if not items:
                continue
            values = [self._normalize(item.value) for item in items]
            lines.append(f"{slot.spl_prefix}[{','.join(values)}]")
        return lines

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().replace(" ", "_")
