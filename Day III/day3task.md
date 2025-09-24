📝 Day 3 Exercise Sheet: Advanced Memory, RAG & Tools with CrewAI
🎯 Goal
Design a multi-agent workflow that uses versioned memory, an advanced RAG pipeline, and tool use (browser/code/API) while optimizing cost & performance with local SLMs (Ollama).

🧪 What You’ll Learn

How to wire short/long/entity memory in CrewAI and keep versioned knowledge for reproducibility

How to build an advanced RAG stack (ingest → embed → retrieve → ground generations) tuned for SLMs

How to attach tools (browser/code executor/API) to agents and control tool use

How to apply performance optimizations (batching, caching, streaming) with local models

📋 Instructions
1) Prepare data & memory

Create a small /data/ folder with 8–12 short docs (Markdown, TXT, or PDFs). Mix reference notes and task specs.

Initialize a vector store (choose one):

Chroma (local) – simplest (default embeddings via Ollama/nomic-embed or all-MiniLM alt)

Postgres + pgvector (advanced) – if you already have it running

Define memory strategy per agent:

short_memory=True, long_memory=True, entity_memory=True

Add a simple memory tag (e.g., v1.0-day3) to every write so you can “pin” a version later.

2) Build the RAG pipeline

Ingest the /data/ docs → split (by paragraphs/headings) → embed → upsert into your store.

Implement retrieve(query, k=5, filter={"tag":"v1.0-day3"}) to pull the versioned context.

Create a helper grounded_prompt(task_desc, context_chunks) that:

shows a brief set of rules (be concise, cite chunk IDs you used)

injects top-k chunks with IDs and titles

instructs the model to refuse if context is insufficient (safety net).

3) Define 3–4 agents with tools

Use the same structure as Day 1–2, but add tools here. Example roles:

Planner – decomposes tasks; tool: none (planning only)

Researcher – calls retrieve(); tool: RAGRetriever

Coder – runs code snippets safely; tool: CodeRunner (a constrained exec or mocked function)

Reviewer – checks claims against retrieved chunks; tool: RAGRetriever, Browser (optional)

Each agent: llm="ollama/<model>" (try mixing phi3:mini, mistral, deepseek-r1:1.5b).

4) Create 5–6 tasks with grounding requirements

For each task define:

description (must be solvable using your /data/ docs)

requires_context=True/False

complexity: 1–5

tooling: which tools are allowed

memory_tag: default v1.0-day3 (so you can switch to v1.1 later)

Examples:

Summarize the design notes into a one-pager with bullet citations (ctx required)

Generate a checklist from policy docs (ctx required)

Write a small data parser (code allowed; ctx optional)

Draft a review that validates claims against chunk IDs (ctx required)

Plan next steps with rationale and which chunks are missing (ctx optional)

5) Routing with context-aware selection

Implement choose_best_agent(task, agents) that considers:

role → task type match

requires_context → prefer agents with RAGRetriever

complexity → prefer stronger model for ≥4

fallback to a default agent on tie

Tip: You can still include a short LLM prompt (like Day 2) to justify the selection, but now feed it the task metadata plus whether RAG is available.

6) Wire grounded Task() execution

Before each task, if requires_context, call retrieve() and build grounded_prompt(...).

Construct Task() with expected_output rules:

If grounded → require citing chunk IDs used

If code task → require runnable snippet and brief usage notes

Run with Process.sequential first; optionally try parallel once it works.

7) Add performance optimizations

Enable LLM response streaming where available (faster UX).

Cache retrieval results by (task_hash, memory_tag) to avoid repeated queries.

Batch related retrieval calls (1 embedding call for multiple queries if your stack supports it).

Log token counts / time per step (simple time.perf_counter() + size of prompts).

8) Versioned memory experiment

Duplicate one doc, tweak content (e.g., fix a date), re-ingest as v1.1-day3.

Rerun a task with memory_tag="v1.1-day3" and compare results vs v1.0-day3.

Note differences and whether the Reviewer flags inconsistencies.

9) Visualize & log

Print a Markdown report:

Task → chosen agent → tools used

Retrieval chunk IDs cited

Timing + token (approx)

Short reflection: “What would improve accuracy?”

Save a small CSV/JSON: task_id, agent, complexity, used_context, chunk_ids, duration_ms.

🧱 Minimal Template (sketch)
# Pseudocode-level sketch (adapt to your notebook)
from crewai import Agent, Task, Crew, Process

# 1) Vector store + ingest (write your own helpers)
store = MyVectorStore(path="./rag_index")
ingest_docs("./data", store, tag="v1.0-day3")

def retrieve(q, k=5, tag="v1.0-day3"):
    return store.search(q, k=k, filter={"tag": tag})

def grounded_prompt(desc, chunks):
    # return f"RULES...\nCONTEXT:\n{format_chunks(chunks)}\nTASK:\n{desc}\n"
    ...

# 2) Agents with tools
researcher = Agent(role="Researcher", goal="Find & inject ground truth",
                   tools=[RAGRetriever(retrieve)], llm="ollama/phi3:mini",
                   memory=True)
coder = Agent(role="Coder", goal="Implement small utilities",
              tools=[CodeRunner()], llm="ollama/mistral", memory=True)
planner = Agent(role="Planner", goal="Decompose tasks", llm="ollama/phi3:mini", memory=True)
reviewer = Agent(role="Reviewer", goal="Validate claims vs context",
                 tools=[RAGRetriever(retrieve)], llm="ollama/deepseek-r1:1.5b", memory=True)

agents = [planner, researcher, coder, reviewer]

def choose_best_agent(task, agents):
    # use task.requires_context, complexity, role matching; optionally an LLM tie-breaker
    ...

# 3) Tasks
tasks = [
  {"description":"Summarize design notes with citations", "requires_context":True, "complexity":3, "tooling":["RAGRetriever"], "memory_tag":"v1.0-day3"},
  ...
]

crew_tasks = []
for t in tasks:
    agent = choose_best_agent(t, agents)
    desc = t["description"]
    if t["requires_context"]:
        ctx = retrieve(desc, k=5, tag=t["memory_tag"])
        desc = grounded_prompt(desc, ctx)
    crew_tasks.append(Task(description=desc, expected_output="Cite chunk IDs if used", agent=agent))

crew = Crew(agents=agents, tasks=crew_tasks, process=Process.sequential)
results = crew.kickoff()

# 4) Reporting
print_markdown_report(results, logs, timings)  # your helper
save_run_csv(results, "./runs/day3_results.csv")

💡 Optional Challenges

Guardrails: if the model doesn’t cite chunk IDs when required → auto-retry w/ stronger instruction

Retriever re-rank: implement a second-pass relevance re-ranker (e.g., keyword overlap or cosine margin)

Tool policies: prevent Coder from running network calls; Reviewer must always call retriever first

Batching: group similar queries into a single embedding request to cut time

✅ Submission Checklist

You ingested docs and retrieved context with a version tag

You defined 3–4 agents with tools and memory

You implemented context-aware routing and grounded prompts

You ran the Crew and captured results + timings

You produced a Markdown report (and a small CSV/JSON log)

(Optional) You compared outputs across v1.0 vs v1.1 memory tags

🧠 Tip
Today’s focus is quality + efficiency. Ground every answer in retrieved chunks, keep memory versioned for reproducibility, and measure what matters (latency, token size, retries). This mirrors real production agent stacks from research to SMEs: auditable outputs + fast iteration.
(For alignment with the Day 3 plan—advanced memory, RAG, tools, and optimization—see the bootcamp agenda.)