# 🧠 Day II — Multi-Agent Systems & Bidding  
**AI Agents Bootcamp 2025**  
📍 UTM — Chișinău • 📅 24 June 2025  

---

## 🚀 What You’ll Learn

On Day 2, we move from single-agent tasks to collaborative and competitive **multi-agent systems (MAS)**.

You will:
- Define multiple AI agents with unique roles and goals
- Implement **dynamic task allocation** using CrewAI and Ollama
- Simulate coordination via **bidding, reasoning, or role-matching**
- Apply MAS logic to both technical and small business workflows

---

## 📁 Contents

| File | Description |
|------|-------------|
| [`Multi_Agent_Bidding.ipynb`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/Multi_Agent_Bidding.ipynb) | Core notebook: run local agents with task assignment |
| [`2.1.task-bidding.md`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/2.1.task-bidding.md) | Exercise sheet for hands-on practice |
| [`day2_agents.json`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/day2_agents.json) | Predefined agent specifications |
| [`day2_tasks.json`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/day2_tasks.json) | Basic starter task set |
| [`day2_tasks_detailed.json`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/day2_tasks_detailed.json) | Richer, real-world task examples |
| [`Day-2-Multi-Agent-Systems-and-Bidding.pptx`](https://github.com/vipbasil/aibootcamp/blob/main/Day%20II/Day-2-Multi-Agent-Systems-and-Bidding.pptx) | Slides for theory + MAS architectures |

---

## 🧪 How It Works

This MAS system is built using:

- **CrewAI** — Agent coordination framework  
- **Ollama** — Local LLM server (e.g. DeepSeek, Phi3, TinyLLaMA)  
- **Task bidding** — Agents choose or are chosen based on prompt responses  

Use the LLM to ask:  
> "Which agent is best for this task?"  

Then pass the response to a CrewAI task object, and let the team execute.

---

## 📋 Example Use Cases

- 🤖 Software: Planner, Coder, Reviewer  
- 🛍️ Small Business: SalesBot, Bookkeeper, Product Support  
- 📊 Finance: Cost analyzer, Invoice checker, Anomaly detector  

---

## ✅ Checklist Before Running

- [ ] Install Ollama and pull models (`phi3:mini`, `deepseek-r1:7b`, etc.)  
- [ ] Start Ollama locally (`ollama serve`)  
- [ ] Run the notebook from top to bottom  
- [ ] Review agent assignments in the console/log  
- [ ] Try alternate models or task prompts!

---

## 📣 Feedback

Please fill out the short [Day 2 Feedback Form](https://forms.gle/your-feedback-link) after you complete the day.  
It helps us improve and evolve the Bootcamp experience.

---

## 🧠 Coming Next

**Day 3** dives into advanced tools, agent memory, and performance optimization. You’ll learn about:
- RAG pipelines for agents  
- Tool execution (e.g., search, browser)  
- Long-term memory in CrewAI  

📅 See you tomorrow!
