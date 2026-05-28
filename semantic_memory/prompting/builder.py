from semantic_memory.prompting.spl import SPLEncoder


SYSTEM_PROMPT = """You are a reasoning assistant. Context arrives in Symbolic Prompt Language (SPL).
Read the structure directly. Do not invent missing facts.

Core slots: task= deadline= pref=[] !pref=[] ent=[] q_hist= hw= scope= status=
Coding extension slots: stack=[] err=[]
Medical extension slots: sym=[] vitals=[]
Legal extension slots: juris=[] clause=[]

All memories presented are active and verified. Answer precisely based only on what is provided."""


class PromptBuilder:
    def __init__(self):
        self.encoder = SPLEncoder()
        self.system_prompt = SYSTEM_PROMPT

    def build(
        self,
        smos,
        query: str,
        schema_name: str = "general",
        contextual=None,
    ) -> str:
        return self.encoder.encode(
            smos,
            query,
            schema_name=schema_name,
            contextual=contextual,
        )
