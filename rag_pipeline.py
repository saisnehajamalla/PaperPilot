"""Grounded question answering: retrieval -> LLM -> independently-verified citations.

The LLM is trusted only for the answer text and for choosing which SOURCE_n labels
support it. Page, section, paper name, and evidence text always come from our own
retrieved chunk metadata — never from the model's output.
"""

import json
from dataclasses import dataclass, field

from citation import Source, build_source, strip_source_labels
from context_builder import build_context
from llm import LLMClient
from retrieval_pipeline import RetrievalPipeline

REFUSAL_SENTENCE = "I couldn't find this information in the uploaded paper."

SYSTEM_PROMPT = f"""You are PaperPilot, an AI research-paper assistant.

Answer the user's question using only the provided excerpts from the uploaded research paper. Each excerpt is labeled with a source id such as SOURCE_1, SOURCE_2, and so on.

Do not use outside knowledge. Do not invent facts, page numbers, sections, quotations, results, datasets, or citations — page and section metadata is attached separately by the application, not by you.

If the provided excerpts do not contain enough information to answer the question, set "answer" to exactly this sentence: "{REFUSAL_SENTENCE}" and set "source_ids" to an empty list.

Distinguish between information explicitly stated in the excerpts and reasonable synthesis across multiple excerpts; do not go beyond what the excerpts support.

Before answering, inspect all of the retrieved excerpts relevant to the question, not just the first one. If you cite more than one source id, the answer must reflect the substantive information from every source id you cite — do not pick one source and ignore the rest. Conversely, do not cite a source id whose content you are not actually using in the answer, and do not force in an unrelated source just because it was retrieved. For questions asking for a list of facts (e.g. datasets used, methods, metrics, contributions, limitations, results), make the answer complete with respect to the supplied excerpts: if more than one such fact is supported by the excerpts, include all of them. Do not add any fact that the excerpts do not support.

The "answer" text must read as plain prose with no citation markers of any kind in it — never write "SOURCE_1", "Source IDs:", "(see SOURCE_3)", footnote numbers, or anything referring to a source id inside the answer text. Source ids belong ONLY in the separate "source_ids" field.

Respond ONLY with valid JSON matching this schema:
{{"answer": string, "source_ids": array of the SOURCE_n labels (strings) whose excerpts support the answer}}"""


@dataclass
class RAGResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    refused: bool = False
    reason: str = "answered"  # answered | retrieval_refused | llm_refused | ungrounded | llm_error | malformed_response


def _parse_llm_json(content: str) -> tuple[str, list[str]] | None:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    source_ids = data.get("source_ids", [])
    if not isinstance(answer, str):
        return None
    if not isinstance(source_ids, list) or not all(isinstance(s, str) for s in source_ids):
        return None
    return answer, source_ids


class RAGPipeline:
    def __init__(self, retrieval_pipeline: RetrievalPipeline, llm_client: LLMClient):
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client

    def answer(self, question: str) -> RAGResult:
        retrieval = self.retrieval_pipeline.retrieve(question)
        if not retrieval.accepted:
            return RAGResult(answer=REFUSAL_SENTENCE, refused=True, reason="retrieval_refused")

        context, source_map = build_context(retrieval.reranked)
        user_prompt = f"Question: {question}\n\nExcerpts:\n\n{context}"
        llm_response = self.llm_client.complete_json(SYSTEM_PROMPT, user_prompt)

        if not llm_response.ok:
            return RAGResult(
                answer=f"The AI model could not be reached right now ({llm_response.error}). "
                       "Please try again in a moment.",
                refused=True,
                reason="llm_error",
            )

        parsed = _parse_llm_json(llm_response.content)
        if parsed is None:
            return RAGResult(
                answer="The AI model returned a response in an unexpected format. Please try asking again.",
                refused=True,
                reason="malformed_response",
            )

        answer_text, raw_source_ids = parsed
        answer_text = strip_source_labels(answer_text)
        if not answer_text.strip():
            # The model returned a technically-valid but empty answer (observed rarely in
            # practice) — treat it the same as a malformed response rather than showing a
            # blank answer bubble with sources attached to nothing.
            return RAGResult(
                answer="The AI model returned an empty response. Please try asking again.",
                refused=True,
                reason="malformed_response",
            )
        valid_source_ids = [sid for sid in raw_source_ids if sid in source_map]

        if answer_text.strip() == REFUSAL_SENTENCE:
            return RAGResult(answer=REFUSAL_SENTENCE, refused=True, reason="llm_refused")

        if not valid_source_ids:
            # The model gave an answer but cited nothing we can verify — treat it as
            # ungrounded rather than display an uncited claim.
            return RAGResult(answer=REFUSAL_SENTENCE, refused=True, reason="ungrounded")

        sources = [build_source(sid, source_map[sid]) for sid in valid_source_ids]
        return RAGResult(answer=answer_text, sources=sources, refused=False, reason="answered")
