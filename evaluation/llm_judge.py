"""LLM-as-judge scoring with heuristic fallbacks.

When a LangChain chat model is injected the judge uses structured LLM prompts.
Without one it falls back to token-overlap heuristics so experiments can run
fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    """Tokenize text into a normalized set for heuristic scoring."""
    return {tok for tok in re.findall(r"\w+", text.lower()) if tok}


def _overlap_score(reference: str, candidate: str) -> float:
    """Compute token-overlap recall: |ref ∩ cand| / |ref|."""
    ref_tokens = _token_set(reference)
    cand_tokens = _token_set(candidate)
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


# ---------------------------------------------------------------------------
# LLM judge prompts
# ---------------------------------------------------------------------------

_CORRECTNESS_PROMPT = """\
You are an evaluation assistant. Rate how well the generated answer matches the ground truth answer.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Score from 0.0 to 1.0 where:
  1.0 = semantically equivalent, all key facts present
  0.5 = partially correct, some key facts missing
  0.0 = incorrect or completely irrelevant

Reply with only a decimal number between 0.0 and 1.0."""

_FAITHFULNESS_PROMPT = """\
You are an evaluation assistant. Rate how faithful the generated answer is to the provided context.
A faithful answer only states facts present in the context and does not hallucinate.

Context:
{context}

Generated Answer: {answer}

Score from 0.0 to 1.0 where:
  1.0 = every claim in the answer is supported by the context
  0.5 = some claims are unsupported or contradict the context
  0.0 = answer is entirely hallucinated or contradicts the context

Reply with only a decimal number between 0.0 and 1.0."""

_PRECISION_PROMPT = """\
You are an evaluation assistant. Rate the precision of the retrieved contexts for answering the question.
Precision measures what fraction of the retrieved chunks are actually relevant.

Question: {question}
Retrieved Contexts:
{contexts}

Score from 0.0 to 1.0 where:
  1.0 = all retrieved chunks are relevant to answering the question
  0.5 = about half are relevant
  0.0 = none are relevant

Reply with only a decimal number between 0.0 and 1.0."""

_RECALL_PROMPT = """\
You are an evaluation assistant. Rate how well the retrieved contexts cover the information needed to produce the ground truth answer.
Recall measures whether the supporting facts were actually retrieved.

Ground Truth: {ground_truth}
Retrieved Contexts:
{contexts}

Score from 0.0 to 1.0 where:
  1.0 = the retrieved contexts contain all information needed to produce the ground truth
  0.5 = about half of the needed information is present
  0.0 = the retrieved contexts contain no relevant information

Reply with only a decimal number between 0.0 and 1.0."""


def _parse_llm_score(response: str) -> float:
    """Extract the first float in [0, 1] from an LLM response string."""
    match = re.search(r"\d+\.?\d*", response)
    if match:
        raw = float(match.group())
        return max(0.0, min(1.0, raw))
    return 0.0


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

@dataclass
class JudgeScore:
    """Structured score returned by the judge."""

    score: float
    rationale: str


class LLMJudge:
    """Score answer and retrieval quality.

    When llm_client is provided all four metrics use LLM-based evaluation.
    Otherwise the judge falls back to token-overlap heuristics, which are
    reproducible and offline but less semantically accurate.

    Args:
        llm_client: Optional LangChain BaseChatModel. If None, heuristics are used.
    """

    def __init__(self, llm_client=None) -> None:
        """Initialize the judge with an optional LangChain chat model."""
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_answer_correctness(
        self, query: str, ground_truth: str, answer: str
    ) -> JudgeScore:
        """Score semantic agreement between the generated answer and ground truth.

        Args:
            query: Original user question (used by LLM judge for context).
            ground_truth: Reference answer.
            answer: Generated answer to evaluate.
        """
        if self.llm_client is not None:
            return self._llm_score(
                _CORRECTNESS_PROMPT.format(
                    question=query, ground_truth=ground_truth, answer=answer
                ),
                "LLM correctness judge",
            )
        score = _overlap_score(ground_truth, answer)
        return JudgeScore(score=score, rationale="Token overlap between answer and ground truth.")

    def score_faithfulness(self, answer: str, contexts: list[str]) -> JudgeScore:
        """Score whether the answer is grounded in retrieved contexts.

        Args:
            answer: Generated answer.
            contexts: Retrieved text chunks used to generate the answer.
        """
        if self.llm_client is not None:
            combined = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            return self._llm_score(
                _FAITHFULNESS_PROMPT.format(context=combined, answer=answer),
                "LLM faithfulness judge",
            )
        combined_context = " ".join(contexts)
        score = _overlap_score(answer, combined_context)
        return JudgeScore(score=score, rationale="Answer terms traced against retrieved context.")

    def score_context_precision(self, query: str, contexts: list[str]) -> JudgeScore:
        """Score what fraction of retrieved contexts are relevant to the query.

        Args:
            query: Original user question.
            contexts: Retrieved text chunks.
        """
        if not contexts:
            return JudgeScore(score=0.0, rationale="No contexts retrieved.")
        if self.llm_client is not None:
            formatted = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            return self._llm_score(
                _PRECISION_PROMPT.format(question=query, contexts=formatted),
                "LLM context precision judge",
            )
        per_context = [_overlap_score(query, context) for context in contexts]
        score = sum(per_context) / len(per_context)
        return JudgeScore(score=score, rationale="Average query overlap per context.")

    def score_context_recall(self, ground_truth: str, contexts: list[str]) -> JudgeScore:
        """Score how much of the ground truth is covered by retrieved contexts.

        Args:
            ground_truth: Reference answer.
            contexts: Retrieved text chunks.
        """
        if self.llm_client is not None:
            formatted = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            return self._llm_score(
                _RECALL_PROMPT.format(ground_truth=ground_truth, contexts=formatted),
                "LLM context recall judge",
            )
        combined_context = " ".join(contexts)
        score = _overlap_score(ground_truth, combined_context)
        return JudgeScore(score=score, rationale="Ground-truth coverage in retrieved context.")

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------

    def _llm_score(self, prompt: str, rationale: str) -> JudgeScore:
        """Invoke the LLM and parse a 0–1 score from its response."""
        try:
            from langchain_core.messages import HumanMessage

            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            score = _parse_llm_score(text)
            return JudgeScore(score=score, rationale=f"{rationale} (raw: {text[:80]})")
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_judge_failed", error=str(exc), fallback="heuristic")
            return JudgeScore(score=0.0, rationale=f"LLM judge error: {exc}")
