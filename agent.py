"""Your capstone agent: the one your README demos and your CI grades.

It starts as the final assignment's starter, unchanged: the same `YourAgent`,
the same `answer_question` pipeline from the course package, the same budget.
Calling it returns a `bootcamp_agent.schema.ResearchAnswer`, the contract the
whole course used, so everything you built in the sessions plugs in here.
`run(question)` returns the whole `AgentResult`, trace included, which is what
`uv run bootcamp capstone trace "<question>"` prints.

As shipped it is honest and insufficient. On the offline `FakeLLM` it refuses
what it should refuse and answers nothing else, and some contract tests in
`tests/test_contract.py` are marked as expected failures on purpose. Making them
pass is the work. What to add, session by session, is in `docs/` (each file
names the session that fills it).

The provider comes from `.env` (`BOOTCAMP_PROVIDER`), and falls back to the
offline `FakeLLM`. Keys live only in `.env`, which git ignores.
"""

from __future__ import annotations

from pathlib import Path

from bootcamp_agent.agent import AgentResult, answer_question
from bootcamp_agent.config import load_settings
from bootcamp_agent.documents import Document, load_corpus
from bootcamp_agent.llm import LLMClient, get_client
from bootcamp_agent.schema import ResearchAnswer
from bootcamp_agent.tools import Tool, build_tools
class StrictCitationClient:
    """Adds stricter citation discipline without changing the provider."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def complete(self, system: str, user: str) -> str:
        stricter_system = (
            system
            + "\n\nCITATION RULES:\n"
            + "- Every citation MUST be copied verbatim from a document ID shown "
              "inside square brackets in the Context.\n"
            + "- NEVER invent citation names such as doc-1, doc-2, source-1, "
              "document-1, or similar placeholders.\n"
            + "- Cite ONLY documents that directly support the claims in the answer.\n"
            + "- If several retrieved passages have the same document ID, cite that "
              "document ID only once.\n"
            + "- Do not cite every retrieved document automatically.\n"
            + "- Answer the user's question directly and completely using the most relevant retrieved passage.\n"
            + "- Prefer a document whose topic directly matches the question over tangentially related documents.\n"
            + "- If the relevant context explicitly lists multiple items, include ALL relevant items from that list.\n"
            + "- Do not replace a directly relevant answer with related information from another document.\n"
            + "- Keep the citation list minimal: include only documents that are necessary to support the final answer.\n"
            + "- If one document fully supports the answer, cite ONLY that document even if other retrieved documents are related.\n"
            + "- Preserve the key terminology and causal statements from the most relevant context; do not paraphrase away important conditions, contrasts, or consequences.\n"
            + "- For WHY or HOW questions, explicitly state the reason, the mechanism, and the practical consequence when the context provides them.\n"
            + "- When the context presents a named list or layered set of recommendations, include the complete set and the main conclusion that introduces or explains it.\n"
        )
        return self.client.complete(system=stricter_system, user=user)
#: The six course documents, copied in by `bootcamp capstone new`. Versioned
#: input: nothing you build writes to it.
CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"


class YourAgent:
    """The agent the tests and the grader run. Make it yours."""

    #: How long one provider call may take before the agent gives up with a
    #: flagged refusal. NOT ENFORCED YET: the starter waits for ever, which is
    #: why the `timeout` contract test is marked xfail. The test sets this low
    #: and expects an answer inside a second.
    timeout_s: float = 30.0

    def __init__(self, client: LLMClient | None = None) -> None:
        self.documents: list[Document] = load_corpus(CORPUS_DIR)
        base_client = client if client is not None else get_client(load_settings())
        self.client = StrictCitationClient(base_client)        # Every tool the agent can reach. Session 4's registry, read-only by
        # construction; session 12 has you classify each one, and the `tools`
        # contract test refuses anything not classified as a reader.
        self.tools: dict[str, Tool] = build_tools(self.documents, self.client)
    def _select_documents(self, question: str) -> list[Document]:
        """Prefer documents whose topic directly matches the question."""
        q = question.lower()

        # Prompt-injection questions must also keep injected test documents,
        # so match this topic by content as well as canonical document id.
        if "prompt injection" in q or "injection" in q:
            matches = []
            for doc in self.documents:
                searchable = " ".join(
                    (
                        doc.doc_id,
                        doc.title,
                        " ".join(doc.tags),
                        doc.text,
                    )
                ).lower()

                if "prompt injection" in searchable or "injection" in searchable:
                    matches.append(doc)

            if matches:
                return matches

        # The remaining topics have a canonical corpus document.
        topic_documents = (
            (("structured output",), "structured-outputs"),
            (("golden evaluation", "golden set"), "evaluation-basics"),
            (("retrieval-augmented", "chunking", " rag "), "rag-basics"),
            (("stopping condition", "agent loop", "tool-call budget"), "agent-loops"),
            (("mcp",), "mcp-overview"),
        )

        padded_q = f" {q} "

        for hints, doc_id in topic_documents:
            if any(hint in padded_q for hint in hints):
                matches = [doc for doc in self.documents if doc.doc_id == doc_id]
                if matches:
                    return matches

        return self.documents
        
    def run(self, question: str) -> AgentResult:
        """One question, answered or refused, with the trace of how."""
        selected_documents = self._select_documents(question)

        result = answer_question(
            question,
            selected_documents,
            self.client,
            max_tool_calls=3,
            top_k=5,
        )

        answer = result.answer

        if (
            answer.needs_human_review
            and not answer.citations
            and answer.confidence <= 0.2
        ):
            answer = ResearchAnswer(
                answer="I don't know based on the provided corpus.",
                citations=(),
                confidence=0.0,
                needs_human_review=True,
            )
            return AgentResult(answer=answer, trace=result.trace)

        return result

    def __call__(self, question: str) -> ResearchAnswer:
        return self.run(question).answer