import os
import sys
import json
import time
import secrets
import requests
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/app")
sys.path.insert(0, "/content")

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nsagent.db_retriever import DatabaseRetriever
from nsagent.expert import PythonExpertAgent


WALLET_ADDRESS = os.environ.get(
    "WALLET_ADDRESS", "0x12133a4f996bdc9d2894b441e1f8621b499d2c3c"
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

app = FastAPI(
    title="Neuro-Symbolic Python Agent API",
    version="5.0.0",
    description="Self-trained retrieval-first Python expert API.",
)


class AskPythonRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class GenerateScriptRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=500)


class CodebaseQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AgentResponse(BaseModel):
    request_type: str
    message: str
    trace: List[str] = []
    output: Optional[str] = None
    tool: Optional[str] = None
    status: str = "ok"


_expert: Optional[PythonExpertAgent] = None
_trial_keys: Dict[str, Dict[str, Any]] = {}
_paid_keys: Dict[str, Dict[str, Any]] = {}


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _sb_post(table, payload):
    if not SUPABASE_URL:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text[:200])


def _sb_get(table, filters=None):
    if not SUPABASE_URL:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, headers=_sb_headers(), params=filters or {}, timeout=30)
    if r.status_code >= 400:
        return []
    return r.json()


def _sb_patch(table, payload, filters):
    if not SUPABASE_URL:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.patch(url, headers=_sb_headers(), json=payload, params=filters, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text[:200])


def get_expert() -> PythonExpertAgent:
    global _expert
    if _expert is None:
        db_path = os.environ.get("PYTHON_KNOWLEDGE_DB", "/app/agent_data/python_knowledge.db")
        skills_db = os.environ.get("SKILLS_DB", "/app/agent_data/full_training_runs.db")
        _expert = PythonExpertAgent(
            db_path=db_path,
            skills_db_path=skills_db,
            project_root=os.environ.get("AGENT_PROJECT_ROOT", "/app/sample_project"),
        )
    return _expert


def require_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    if x_api_key == "demo-key":
        return x_api_key

    paid_env = os.environ.get("PAID_API_KEYS", "")
    if x_api_key in [k.strip() for k in paid_env.split(",") if k.strip()]:
        return x_api_key

    if SUPABASE_URL:
        rows = _sb_get("trial_keys", {
            "api_key": f"eq.{x_api_key}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if rows:
            row = rows[0]
            used = int(row.get("quota_used", 0))
            limit = int(row.get("quota_limit", 50))
            if used >= limit:
                raise HTTPException(status_code=429, detail="Free trial quota exhausted.")
            _sb_patch("trial_keys", {"quota_used": used + 1}, {"api_key": f"eq.{x_api_key}"})
            return x_api_key

        expired = _sb_get("trial_keys", {"api_key": f"eq.{x_api_key}"})
        if expired:
            raise HTTPException(status_code=401, detail="Trial expired.")

        paid_rows = _sb_get("paid_keys", {
            "api_key": f"eq.{x_api_key}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if paid_rows:
            return x_api_key

        any_paid = _sb_get("paid_keys", {"api_key": f"eq.{x_api_key}"})
        if any_paid:
            raise HTTPException(status_code=401, detail="API key expired.")

    if x_api_key in _trial_keys:
        t = _trial_keys[x_api_key]
        if time.time() > t["expires_at"]:
            raise HTTPException(status_code=401, detail="Trial expired.")
        if t["quota_used"] >= t["quota_limit"]:
            raise HTTPException(status_code=429, detail="Trial quota exhausted.")
        t["quota_used"] += 1
        return x_api_key

    if x_api_key in _paid_keys:
        p = _paid_keys[x_api_key]
        if time.time() > p["expires_at"]:
            raise HTTPException(status_code=401, detail="API key expired.")
        return x_api_key

    raise HTTPException(status_code=401, detail="Invalid API key.")


@app.post("/v1/billing/free-trial")
def free_trial(request: Request):
    ip = request.headers.get("x-forwarded-for", "")
    ip = ip.split(",")[0].strip() if ip else (request.client.host if request.client else "unknown")

    if SUPABASE_URL:
        rows = _sb_get("trial_keys", {
            "ip_address": f"eq.{ip}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if rows:
            row = rows[0]
            return {
                "status": "trial_already_exists",
                "api_key": row["api_key"],
                "expires_in_days": int(os.environ.get("TRIAL_DAYS", "7")),
                "quota_limit": int(row.get("quota_limit", 50)),
                "message": "Returning your existing active trial key.",
            }

        recent = _sb_get("trial_keys", {"ip_address": f"eq.{ip}"})
        if recent:
            return {"status": "trial_already_used", "message": "Free trial already used from this IP."}

    key = "sk_trial_" + secrets.token_urlsafe(24)
    days = int(os.environ.get("TRIAL_DAYS", "7"))
    quota_limit = int(os.environ.get("TRIAL_QUOTA", "50"))
    expires_at = int(time.time()) + (days * 86400)

    if SUPABASE_URL:
        _sb_post("trial_keys", {
            "api_key": key,
            "email": "free-trial",
            "expires_at": expires_at,
            "quota_limit": quota_limit,
            "quota_used": 0,
            "ip_address": ip,
        })
    else:
        _trial_keys[key] = {
            "expires_at": expires_at,
            "quota_limit": quota_limit,
            "quota_used": 0,
        }

    return {
        "status": "trial_created",
        "api_key": key,
        "expires_in_days": days,
        "quota_limit": quota_limit,
        "usage": "Use X-API-Key header on /v1/ask-python and /v1/generate-script.",
    }


@app.get("/v1/billing/crypto/direct-address")
def direct_evm_address(chain: str = "polygon"):
    return {
        "request_id": secrets.token_hex(8),
        "wallet_address": WALLET_ADDRESS,
        "chain": chain,
        "symbol": "POL",
        "amount": float(os.environ.get("SUBSCRIPTION_PRICE_POL", "469")),
        "billing": "monthly",
    }


@app.get("/v1/billing/crypto/verify")
def verify_payment(tx_hash: str, request_id: str):
    return {
        "status": "not_implemented",
        "message": "On-chain verification currently offline in minimal build.",
    }


def _to_response(result: Dict[str, Any], request_type: str) -> AgentResponse:
    rtype = result.get("type", "unknown")
    trace = result.get("trace", []) or []
    msg = result.get("message", "")

    if rtype == "python_script":
        gen = result.get("generated")
        er = result.get("exec_result")
        out = None
        if er is not None and hasattr(er, "stdout"):
            out = (er.stdout or "").strip()[:2000]
        return AgentResponse(
            request_type=request_type,
            message=f"Generated: {getattr(gen, 'filename', 'unknown')}",
            trace=trace,
            output=out,
            tool="generate_python_script",
            status="ok" if er is not None and getattr(er, "success", False) else "error",
        )

    if rtype == "answer":
        return AgentResponse(request_type=request_type, message=msg, trace=trace, status="ok")

    return AgentResponse(request_type=request_type, message=msg or str(result), trace=trace, status="ok")


@app.post("/v1/ask-python", response_model=AgentResponse)
def ask_python(req: AskPythonRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    result = get_expert().answer(req.question)
    return _to_response(result, "ask_python")


@app.post("/v1/generate-script", response_model=AgentResponse)
def generate_script(req: GenerateScriptRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    result = get_expert().generate(req.task)
    return _to_response(result, "generate_script")


@app.post("/v1/codebase-query", response_model=AgentResponse)
def codebase_query(req: CodebaseQueryRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    result = get_expert().answer(req.question)
    return _to_response(result, "codebase_query")


@app.get("/")
def root():
    return {"service": "python-agent-api", "docs": "/docs", "trial": "/v1/billing/free-trial"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "neuro-symbolic-python-agent"}
