
"""
nsagent_api.main — FastAPI wrapper for the neuro-symbolic Python agent.

Endpoints:
  POST /v1/ask-python          Python knowledge question (stdlib + MBPP)
  POST /v1/generate-script     Generate and run a standalone Python script
  POST /v1/codebase-query      Structural/causal question on the project graph

Every response includes a symbolic trace and all code executes only through
RealSandbox. API-key quota tracking is included; in production replace the
in-memory store with Redis/PostgreSQL + Stripe.
"""
from __future__ import annotations

import os
import sys
import time
import requests
import json
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/content")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from web3 import Web3
from pydantic import BaseModel, Field

from nsagent.agent import NeuroSymbolicAgent

app = FastAPI(
    title="Neuro-Symbolic Python Agent API",
    version="0.1.0",
    description="Offline neuro-symbolic Python coding and codebase intelligence API.",
)

# ---------------------------------------------------------------- agent cache
_AGENT: Optional[NeuroSymbolicAgent] = None

def get_agent() -> NeuroSymbolicAgent:
    global _AGENT
    if _AGENT is None:
        project_root = os.environ.get("AGENT_PROJECT_ROOT", "/app/sample_project")
        state_path = os.environ.get("AGENT_STATE_PATH", "/app/agent_state/final_model_v3.json")
        _AGENT = NeuroSymbolicAgent(
            project_root=project_root,
            state_path=state_path,
        )
    return _AGENT


# ---------------------------------------------------------- quota & API keys
# In production: Redis/PostgreSQL + Stripe. For the Colab demo we use
# a simple in-memory quota store keyed by API key.
FREE_MONTHLY_QUOTA = 100
_quota: Dict[str, Dict[str, int]] = {}
_direct_evm_requests: Dict[str, Dict[str, Any]] = {}
_direct_evm_keys: Dict[str, str] = {}
_direct_evm_key_expiry: Dict[str, float] = {}

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def _sb_post(table, payload):
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase insert error: {r.text[:200]}")

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
        raise HTTPException(status_code=502, detail=f"Supabase update error: {r.text[:200]}")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def _sb_post(table, payload):
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase insert error: {r.text[:200]}")

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
        raise HTTPException(status_code=502, detail=f"Supabase update error: {r.text[:200]}")


def check_quota(api_key: str) -> None:
    now = int(time.time())
    month = now // (30 * 24 * 3600)
    entry = _quota.setdefault(api_key, {"month": month, "used": 0})
    if entry["month"] != month:
        entry["month"] = month
        entry["used"] = 0
    if entry["used"] >= FREE_MONTHLY_QUOTA:
        raise HTTPException(status_code=429, detail="Monthly API quota exceeded.")
    entry["used"] += 1


def require_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if x_api_key == "demo-key":
        check_quota(x_api_key)
        return x_api_key

    paid_env = os.environ.get("PAID_API_KEYS", "")
    paid_keys = [k.strip() for k in paid_env.split(",") if k.strip()]
    if x_api_key in paid_keys:
        return x_api_key

    if SUPABASE_URL:
        rows = _sb_get("paid_keys", {
            "api_key": f"eq.{x_api_key}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if rows:
            return x_api_key

        any_rows = _sb_get("paid_keys", {"api_key": f"eq.{x_api_key}"})
        if any_rows:
            raise HTTPException(status_code=401, detail="API key expired. Pay for another month.")

    if "_direct_evm_keys" in globals() and x_api_key in _direct_evm_keys:
        expiry = _direct_evm_key_expiry.get(x_api_key)
        if expiry and time.time() > expiry:
            raise HTTPException(status_code=401, detail="API key expired. Pay for another month.")
        return x_api_key

    raise HTTPException(status_code=401, detail="Invalid API key.")

def _extract_stdout(result: Any) -> Optional[str]:
    """Best-effort extraction of sandbox stdout from a tool result."""
    try:
        inner = getattr(result, "result", None)
        if inner is None:
            return None
        if isinstance(inner, dict):
            exec_res = inner.get("exec_result")
            if exec_res is not None and hasattr(exec_res, "stdout"):
                return (exec_res.stdout or "").strip()[:2000] or None
            if inner.get("generated"):
                gen = inner["generated"]
                exec_res = inner.get("exec_result")
                if exec_res is not None:
                    return (exec_res.stdout or "").strip()[:2000] or None
        if hasattr(inner, "stdout"):
            return (inner.stdout or "").strip()[:2000] or None
        return None
    except Exception:
        return None


def _to_response(result: Dict[str, Any], request_type: str) -> AgentResponse:
    rtype = result.get("type", "unknown")
    trace = result.get("trace", []) or []
    message = result.get("message", "")

    if rtype == "tool_result":
        tool_res = result.get("result")
        if tool_res is not None:
            msg = getattr(tool_res, "message", "") or str(tool_res)
            message = msg
            output = _extract_stdout(tool_res)
            tool = getattr(tool_res, "tool", None)
            return AgentResponse(
                request_type=request_type,
                message=message,
                trace=trace,
                output=output,
                tool=tool,
                status="ok" if getattr(tool_res, "ok", False) else "error",
            )

    if rtype == "answer":
        return AgentResponse(
            request_type=request_type,
            message=message,
            trace=trace,
            status="ok",
        )

    if rtype == "composite":
        outputs = result.get("outputs", {})
        count = len(outputs)
        return AgentResponse(
            request_type=request_type,
            message=f"Executed {count} subgoals successfully.",
            trace=trace,
            status="ok",
        )

    if rtype == "refusal":
        return AgentResponse(
            request_type=request_type,
            message=message or "Request refused: ungrounded or unsafe.",
            trace=trace,
            status="refused",
        )

    if rtype == "python_script":
        gen = result.get("generated")
        exec_res = result.get("exec_result")
        output = (exec_res.stdout or "").strip() if exec_res is not None else None
        return AgentResponse(
            request_type=request_type,
            message=f"Generated script: {getattr(gen, 'filename', 'unknown')}",
            trace=trace,
            output=output[:2000] if output else None,
            status="ok" if exec_res is not None and getattr(exec_res, "success", False) else "error",
        )

    return AgentResponse(
        request_type=request_type,
        message=message or str(result),
        trace=trace,
        status="ok",
    )


# ------------------------------------------------------------------ endpoints
@app.post("/v1/ask-python", response_model=AgentResponse)
def ask_python(req: AskPythonRequest, api_key: Optional[str] = Header(None, alias="X-API-Key")) -> AgentResponse:
    require_api_key(api_key)
    agent = get_agent()
    result = agent.handle(req.question)
    return _to_response(result, "ask_python")


@app.post("/v1/generate-script", response_model=AgentResponse)
def generate_script(req: GenerateScriptRequest, api_key: Optional[str] = Header(None, alias="X-API-Key")) -> AgentResponse:
    require_api_key(api_key)
    agent = get_agent()
    result = agent.handle(req.task)
    return _to_response(result, "generate_script")


@app.post("/v1/codebase-query", response_model=AgentResponse)
def codebase_query(req: CodebaseQueryRequest, api_key: Optional[str] = Header(None, alias="X-API-Key")) -> AgentResponse:
    require_api_key(api_key)
    agent = get_agent()
    result = agent.handle(req.question)
    return _to_response(result, "codebase_query")


from web3 import Web3
import secrets
import time

WALLET_ADDRESS = "0x12133a4f996bdc9d2894b441e1f8621b499d2c3c"
CHAINS = {"ethereum": {"rpc": "https://eth.llamarpc.com", "chain_id": 1, "decimals": 18, "symbol": "ETH"}, "polygon": {"rpc": "https://polygon.llamarpc.com", "chain_id": 137, "decimals": 18, "symbol": "POL"}, "bsc": {"rpc": "https://bsc-dataseed.binance.org", "chain_id": 56, "decimals": 18, "symbol": "BNB"}, "avalanche": {"rpc": "https://api.avax.network/ext/bc/C/rpc", "chain_id": 43114, "decimals": 18, "symbol": "AVAX"}, "arbitrum": {"rpc": "https://arb1.arbitrum.io/rpc", "chain_id": 42161, "decimals": 18, "symbol": "ARB"}, "optimism": {"rpc": "https://mainnet.optimism.io", "chain_id": 10, "decimals": 18, "symbol": "OP"}, "base": {"rpc": "https://mainnet.base.org", "chain_id": 8453, "decimals": 18, "symbol": "ETH"}}

# Store active payment requests
_direct_evm_requests: Dict[str, Dict[str, Any]] = {}
_direct_evm_keys: Dict[str, str] = {}
_direct_evm_key_expiry: Dict[str, float] = {}

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def _sb_post(table, payload):
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase insert error: {r.text[:200]}")

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
        raise HTTPException(status_code=502, detail=f"Supabase update error: {r.text[:200]}")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def _sb_post(table, payload):
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase insert error: {r.text[:200]}")

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
        raise HTTPException(status_code=502, detail=f"Supabase update error: {r.text[:200]}")


CHAINS = {
    # Mainnets
    "ethereum": {"rpc": "https://eth.llamarpc.com", "chain_id": 1, "decimals": 18, "symbol": "ETH"},
    "polygon":  {"rpc": "https://polygon.llamarpc.com", "chain_id": 137, "decimals": 18, "symbol": "POL"},
    "bsc":      {"rpc": "https://bsc-dataseed.binance.org", "chain_id": 56, "decimals": 18, "symbol": "BNB"},
    "avalanche":{"rpc": "https://api.avax.network/ext/bc/C/rpc", "chain_id": 43114, "decimals": 18, "symbol": "AVAX"},
    "arbitrum": {"rpc": "https://arb1.arbitrum.io/rpc", "chain_id": 42161, "decimals": 18, "symbol": "ARB"},
    "optimism": {"rpc": "https://mainnet.optimism.io", "chain_id": 10, "decimals": 18, "symbol": "OP"},
    "base":     {"rpc": "https://mainnet.base.org", "chain_id": 8453, "decimals": 18, "symbol": "ETH"},

    # Testnets — free tokens
    "polygon-amoy": {"rpc": "https://rpc-amoy.polygon.technology", "chain_id": 80002, "decimals": 18, "symbol": "tMATIC"},
    "ethereum-sepolia": {"rpc": "https://ethereum-sepolia-rpc.publicnode.com", "chain_id": 11155111, "decimals": 18, "symbol": "tETH"},
}

def _make_payment_amount(payment_id: int, decimals: int) -> int:
    """Real payment amount in wei.

    Base amount is read from PAYMENT_AMOUNT_NATIVE env (default 29 MATIC).
    We add the payment id as 6 decimal digits so each invoice is unique.
    """
    amount_native = float(os.environ.get("PAYMENT_AMOUNT_NATIVE", "29"))
    base_wei = int(amount_native * (10 ** decimals))
    micro_native = (payment_id % 1_000_000) / 1_000_000
    micro_wei = int(micro_native * (10 ** decimals))
    return base_wei + micro_wei



def _verify_evm_tx(chain: str, tx_hash: str, expected_wei: int) -> bool:
    cfg = CHAINS.get(chain)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}")
    w3 = Web3(Web3.HTTPProvider(cfg["rpc"]))
    if not w3.is_connected():
        raise HTTPException(status_code=502, detail=f"Could not connect to {chain} RPC")
    try:
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Transaction not found: {exc}")

    if tx["to"].lower() != WALLET_ADDRESS.lower():
        raise HTTPException(status_code=400, detail="Transaction recipient does not match wallet address")
    if tx["value"] != expected_wei:
        raise HTTPException(status_code=400, detail=f"Transaction amount mismatch. Expected {expected_wei} wei, got {tx['value']}")
    return True


@app.get("/v1/billing/crypto/direct-address")
def direct_evm_address(chain: str = "polygon"):
    if chain not in CHAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}. Available: {list(CHAINS)}")

    amount_native = float(os.environ.get("SUBSCRIPTION_PRICE_POL", "469"))
    decimals = CHAINS[chain]["decimals"]
    amount_wei = int(amount_native * (10 ** decimals))

    payment_id = secrets.randbelow(900000) + 100000
    request_id = secrets.token_hex(8)
    subscription_days = int(os.environ.get("SUBSCRIPTION_DAYS", "30"))

    payload = {
        "request_id": request_id,
        "payment_id": payment_id,
        "chain": chain,
        "amount_wei": amount_wei,
        "amount_native": amount_native,
        "wallet": WALLET_ADDRESS,
        "status": "waiting",
    }

    if SUPABASE_URL:
        _sb_post("payment_requests", payload)
    else:
        _direct_evm_requests[request_id] = {
            **payload,
            "created_at": time.time(),
            "subscription_days": subscription_days,
        }

    return {
        "request_id": request_id,
        "payment_id": payment_id,
        "wallet_address": WALLET_ADDRESS,
        "chain": chain,
        "symbol": CHAINS[chain]["symbol"],
        "amount": amount_native,
        "amount_wei": amount_wei,
        "instructions": (
            f"Send EXACTLY {amount_native} {CHAINS[chain]['symbol']} to {WALLET_ADDRESS} "
            f"on {chain}. Then submit the transaction hash."
        ),
        "billing": "monthly",
    }
@app.get("/v1/billing/crypto/verify")
def direct_evm_verify(tx_hash: str, request_id: str):
    if SUPABASE_URL:
        rows = _sb_get("payment_requests", {"request_id": f"eq.{request_id}"})
        if not rows:
            raise HTTPException(status_code=404, detail="Payment request not found")
        req = rows[0]
        if req.get("status") == "paid":
            raise HTTPException(status_code=400, detail="Payment already verified")
    else:
        if request_id not in _direct_evm_requests:
            raise HTTPException(status_code=404, detail="Payment request not found")
        req = _direct_evm_requests[request_id]
        if req["status"] == "paid":
            raise HTTPException(status_code=400, detail="Payment already verified")

    _verify_evm_tx(req["chain"], tx_hash, int(req["amount_wei"]))

    key = "sk_live_" + secrets.token_urlsafe(24)
    days = int(os.environ.get("SUBSCRIPTION_DAYS", "30"))
    expires_at = int(time.time()) + (days * 86400)

    if SUPABASE_URL:
        _sb_post("paid_keys", {
            "api_key": key,
            "email": "crypto-paid-user",
            "expires_at": expires_at,
        })
        _sb_patch("payment_requests", {
            "status": "paid",
            "tx_hash": tx_hash,
        }, {"request_id": f"eq.{request_id}"})
    else:
        req["status"] = "paid"
        _direct_evm_keys[key] = "evm-paid-user"
        _direct_evm_key_expiry[key] = expires_at

    return {
        "status": "paid",
        "api_key": key,
        "request_id": request_id,
        "chain": req["chain"],
        "amount_paid": float(req["amount_native"]),
        "expires_in_days": days,
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at)),
    }
@app.get("/health")
def health():
    return {"status": "ok", "service": "neuro-symbolic-python-agent"}
