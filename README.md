# Neuro-Symbolic Python Agent

A production-grade neuro-symbolic agent that:

- Ingests a real Python project into a typed `NetworkX MultiDiGraph`
- Executes code only inside isolated subprocesses with timeouts
- Grounds every natural-language plan in the symbolic graph
- Answers structural and causal questions with traces
- Diagnoses and patches failing functions via shadow-test-commit loops
- Plans composite goals as a dependency DAG
- Generates standalone Python scripts from natural language using an offline MBPP corpus
- Retrieves Python standard-library documentation via sandboxed `pydoc`
- Accepts crypto payments directly to an EVM wallet and issues paid API keys

## Architecture

| Layer | Implementation |
|---|---|
| Neural perception | spaCy for NL intent + entity extraction |
| Symbolic world model | NetworkX MultiDiGraph with typed edges (`defines`, `calls`, `imports`, `assigns`, `raised`, `thrown_by`, `attempted`, `patched`) |
| Intent compiler | NL → grounded `PlanSpec`; ungrounded entities are refused with clarification |
| Causal memory | Runtime failures stored as edges; `why did X fail?` answered via backward traversal |
| Sandbox | Isolated subprocess execution with timeouts and captured stdout/stderr |
| Self-healing | AST patch → shadow copy → test suite → commit only on green |
| Planner | Composite requests decomposed into a printed DAG of tool calls |
| Knowledge | Cleaned MBPP (967 examples) with TF-IDF + lexical hybrid retrieval |
| Stdlib docs | Sandboxed `pydoc` for arbitrary stdlib modules |
| Billing | Direct EVM crypto payment verification with fallback RPCs |

## Quickstart

### Local

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn nsagent_api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t nsagent .
docker run -p 8000:8000 nsagent
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/ask-python` | Answer a Python knowledge question |
| POST | `/v1/generate-script` | Generate and run a Python script from NL |
| POST | `/v1/codebase-query` | Query the ingested project graph |
| GET | `/v1/billing/crypto/direct-address` | Create a crypto payment request |
| GET | `/v1/billing/crypto/verify` | Verify transaction and issue paid API key |
| GET | `/docs` | Swagger UI |

## Environment Variables

- `AGENT_PROJECT_ROOT` — path to the sample project
- `AGENT_STATE_PATH` — path to the persisted world model JSON
- `PAID_API_KEYS` — comma-separated paid API keys accepted by the agent
- `WALLET_ADDRESS` — EVM wallet that receives crypto payments

## Safety Invariants

- No action without a grounded `PlanSpec`
- No code execution outside the sandbox
- Every answer carries its symbolic trace
- The graph is the source of truth; the neural layer is advisory

## License

MIT
