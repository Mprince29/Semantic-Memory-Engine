from semantic_memory.prompting.spl import SPLEncoder


SYSTEM_PROMPT = """You are a reasoning assistant. Context arrives in Symbolic Prompt Language.
Read the structure directly. Keys may include task, deadline, pref, !pref, ent, q_hist, scope, hw, and status.
Respond precisely and do not invent missing facts."""


class PromptBuilder:
    def __init__(self):
        self.encoder = SPLEncoder()
        self.system_prompt = SYSTEM_PROMPT

    def build(self, smos, query: str) -> str:
        return self.encoder.encode(smos, query)
