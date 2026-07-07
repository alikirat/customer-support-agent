# customer-support-agent

A graph-based AI customer support representative for a shipping company built using **ADK 2.0 (Agent Development Kit)**. The agent dynamically classifies customer queries and routes them to either a Shipping FAQ agent (which answers using a built-in knowledge base with playful emoji responses) or a polite decline node for unrelated queries.

---

## 🛠️ Prerequisites

Before getting started, make sure you have installed:
1. **Python 3.11 or higher**
2. **uv** (Python package manager) - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
3. **agents-cli** - Install using uv:
   ```bash
   uv tool install google-agents-cli
   ```

---

## 🚀 Installation & Setup

1. **Clone the repository** (if not already local) and navigate to the project directory:
   ```bash
   cd customer-support-agent
   ```

2. **Install dependencies**:
   This sets up the virtual environment (`.venv`) and installs the required packages (including `google-adk`):
   ```bash
   agents-cli install
   ```

---

## 🔑 API Key Configuration

To interact with the Gemini models, configure your authentication inside the **`.env`** file.

### Option A: Gemini API (Google AI Studio) - *Recommended*
1. Open the `.env` file in the root directory.
2. Comment out the Vertex AI configurations:
   ```bash
   # GOOGLE_GENAI_USE_VERTEXAI=true
   # GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   # GOOGLE_CLOUD_LOCATION=global
   ```
3. Uncomment and set your API key:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Option B: Vertex AI (Google Cloud Platform)
1. Set the credentials in your local terminal session:
   ```bash
   gcloud auth application-default login
   ```
2. Configure the project ID and location in `.env`:
   ```bash
   GOOGLE_GENAI_USE_VERTEXAI=true
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=global
   ```

---

## 💻 Running the Agent

### 1. Local Playground (Web UI)
Launch the interactive web-based playground to chat with the agent in your browser:
```bash
agents-cli playground
```

### 2. Command Line Interface (CLI)
You can run one-off queries directly in the terminal:
```bash
agents-cli run "How long does standard delivery take?"
```

---

## 🧪 Running Tests & Linting

### Unit Tests
Verify the graph routing logic, state transitions, and node behaviors locally:
```bash
uv run pytest tests/unit
```

### Code Quality & Linting
Run the complete static analysis suite (Ruff formatter, check, Codespell, and Ty type checking):
```bash
agents-cli lint
```

---

## 📁 Project Structure

* **`app/`**: Core agent logic.
  * **`agent.py`**: Defines the nodes, routing, edges, and root `Workflow`.
  * **`fast_api_app.py`**: Local dev server endpoints.
* **`tests/`**: Unit and integration test suites.
* **`pyproject.toml`**: Dependency declarations and linter configurations.
* **`.env`**: Developer configuration & secrets.

---

## Capstone: AI Agents Intensive (Kaggle x Google, June 2026)

This project demonstrates three core concepts from the course:

1. **Multi-Agent ADK 2.0 Graph Workflow** — A graph of specialized nodes (`save_query`, `guardrail`, `classifier_agent`, `router`, `shipping_faq_agent`, `escalate`, etc.) route customer queries to the right handler.

2. **Agent Skill** — An escalation policy skill (`.agents/skills/escalation-policy/`) encodes business rules for when a query should route to a human: expressed anger, legal action mentions, refunds over $200, or repeated queries (3+ times).

3. **Security Guardrail** — A guardrail node intercepts every query before classification, checking for prompt-injection patterns (e.g. "ignore previous instructions," "reveal system prompt"). Flagged queries are short-circuited to a safe generic response instead of reaching the LLM.

All three behaviors are covered by automated unit tests (13 passing) and verified manually via the ADK Playground.

### Visual Walkthrough

![normal-routing](docs/screenshots/normal-routing.png)
*The full agent graph architecture, showing a routine shipping inquiry routed safely through the guardrail, classifier, and router to the shipping FAQ agent.*

![escalation-triggered](docs/screenshots/escalation-triggered.png)
*A query with anger and legal-action language, correctly classified and routed to human escalation.*

![guardrail-flagged](docs/screenshots/guardrail-flagged.png)
*A prompt-injection attempt detected and blocked before reaching the classifier.*
