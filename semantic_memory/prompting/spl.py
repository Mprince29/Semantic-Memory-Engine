from collections import defaultdict

from semantic_memory.domain.models import SemanticMemoryObject


class SPLEncoder:
    type_priority = {
        "task": 1,
        "constraint": 2,
        "preference": 3,
        "entity": 4,
        "event": 5,
        "fact": 6,
    }

    def encode(self, smos: list[SemanticMemoryObject], query: str) -> str:
        grouped: dict[str, list[SemanticMemoryObject]] = defaultdict(list)
        for item in sorted(smos, key=lambda value: self.type_priority.get(value.type, 99)):
            grouped[item.type].append(item)

        lines: list[str] = []
        lines.extend(self._build_task_line(grouped))
        lines.extend(self._build_preference_lines(grouped))
        lines.extend(self._build_fact_lines(grouped))
        lines.extend(self._build_entity_lines(grouped))
        lines.extend(self._build_history_lines(grouped))

        context = "\n".join(lines)
        return f"[CTX]\n{context}\n[/CTX]\n[Q] {query.strip()}"

    @staticmethod
    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75)

    def _build_task_line(self, grouped: dict[str, list[SemanticMemoryObject]]) -> list[str]:
        task_parts: list[str] = []
        for item in grouped.get("task", [])[:3]:
            task_parts.append(f"task={self._normalize(item.value)}")
        for item in grouped.get("constraint", [])[:2]:
            if item.predicate == "deadline":
                task_parts.append(f"deadline={self._normalize(item.value)}")
        return [" ".join(task_parts)] if task_parts else []

    def _build_preference_lines(self, grouped: dict[str, list[SemanticMemoryObject]]) -> list[str]:
        pos = [self._normalize(item.value) for item in grouped.get("preference", []) if "pos" in item.predicate][:4]
        neg = [self._normalize(item.value) for item in grouped.get("preference", []) if "neg" in item.predicate][:4]
        lines: list[str] = []
        if pos:
            lines.append(f"pref=[{','.join(pos)}]")
        if neg:
            lines.append(f"!pref=[{','.join(neg)}]")
        return lines

    def _build_fact_lines(self, grouped: dict[str, list[SemanticMemoryObject]]) -> list[str]:
        facts: list[str] = []
        seen_keys: set[str] = set()
        for item in grouped.get("fact", [])[:4]:
            key = self._normalize(item.predicate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            facts.append(f"{key}={self._normalize(item.value)}")
        return [" ".join(facts)] if facts else []

    def _build_entity_lines(self, grouped: dict[str, list[SemanticMemoryObject]]) -> list[str]:
        entities = [f"{self._normalize(item.subject)}:{self._normalize(item.value)}" for item in grouped.get("entity", [])[:4]]
        return [f"ent=[{','.join(entities)}]"] if entities else []

    def _build_history_lines(self, grouped: dict[str, list[SemanticMemoryObject]]) -> list[str]:
        history: list[str] = []
        seen: set[str] = set()
        for item in grouped.get("event", []):
            token = self._normalize(item.value)
            if token in seen:
                continue
            seen.add(token)
            history.append(token)
            if len(history) == 4:
                break
        return [f"q_hist=[{','.join(history)}]"] if history else []

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().replace(" ", "_")
