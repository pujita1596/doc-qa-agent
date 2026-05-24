from dotenv import load_dotenv
import anthropic
from retriever import retrieve

load_dotenv()

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the internal knowledge base of uploaded documents. "
            "Use this for questions about topics that are likely covered in the documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current events or general information "
            "not found in the documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
]


def search_documents(query: str) -> str:
    chunks = retrieve(query, n_results=3)
    if not chunks:
        return "No relevant documents found."
    return "\n\n".join(
        f"[Source {i + 1}, similarity={c['score']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )


def web_search(query: str) -> str:
    from ddgs import DDGS
    results = DDGS().text(query, max_results=3)
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{r['title']}]\n{r['body']}\nURL: {r['href']}"
        for r in results
    )


SYSTEM_PROMPT = (
    "You are a document Q&A assistant. "
    "Always call search_documents first before answering any question. "
    "Only call web_search if search_documents returns no useful results. "
    "Never answer from memory — ground every claim in tool results."
)


def run_agent(question: str, max_iterations: int = 5) -> dict:
    messages = [{"role": "user", "content": question}]
    plan = []  # track every tool call the agent makes

    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = next(b.text for b in response.content if b.type == "text")
            return {"answer": answer, "plan": plan}

        # stop_reason == "tool_use": Claude wants to call one or more tools
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            plan.append({"tool": block.name, "query": block.input["query"]})
            print(f"  [agent] calling {block.name}({block.input['query']!r})")

            if block.name == "search_documents":
                result = search_documents(block.input["query"])
            elif block.name == "web_search":
                result = web_search(block.input["query"])
            else:
                result = "Unknown tool."

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        # Feed all tool results back so Claude can continue
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "Agent reached max iterations without finishing.", "plan": plan}


if __name__ == "__main__":
    questions = [
        "What is cosine similarity and why is it used in RAG?",  # expects search_documents
        "What is the latest stable version of Python?",           # expects web_search
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = run_agent(q)
        print(f"Plan: {result['plan']}")
        print(f"A: {result['answer']}")
