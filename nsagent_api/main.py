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
from web3 import Web3

from nsagent.db_retriever import DatabaseRetriever
from nsagent.expert import PythonExpertAgent


WALLET_ADDRESS = os.environ.get(
    "WALLET_ADDRESS", "0x12133a4f996bdc9d2894b441e1f8621b499d2c3c"
)

CHAINS = {
    "polygon": {
        "rpcs": [
            "https://polygon-bor-rpc.publicnode.com",
            "https://polygon.drpc.org",
            "https://1rpc.io/matic",
            "https://polygon-rpc.com",
            "https://polygon.llamarpc.com",
            "https://rpc.ankr.com/polygon",
        ],
        "chain_id": 137,
        "decimals": 18,
        "symbol": "POL",
    },
    "polygon-amoy": {
        "rpcs": [
            "https://rpc-amoy.polygon.technology",
            "https://polygon-amoy-bor-rpc.publicnode.com",
            "https://polygon-amoy.g.alchemy.com/v2/demo",
        ],
        "chain_id": 80002,
        "decimals": 18,
        "symbol": "tMATIC",
    },
}


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

app = FastAPI(
    title="Neuro-Symbolic Python Agent API",
    version="6.0.0",
    description="Self-trained Python expert + crypto subscription API.",
)




_expert: Optional[PythonExpertAgent] = None
_trial_keys: Dict[str, Dict[str, Any]] = {}
_paid_keys: Dict[str, Dict[str, Any]] = {}
_direct_evm_requests: Dict[str, Dict[str, Any]] = {}
_direct_evm_key_expiry: Dict[str, float] = {}


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


def _get_working_web3(chain):
    cfg = CHAINS.get(chain)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}")

    for rpc in cfg["rpcs"]:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc))
            if w3.is_connected():
                return w3
        except Exception:
            continue

    raise HTTPException(status_code=502, detail=f"All RPCs for {chain} are unavailable.")


def _verify_evm_tx(chain, tx_hash, expected_wei):
    w3 = _get_working_web3(chain)
    try:
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Transaction not found: {exc}")

    if tx.get("to") and tx["to"].lower() != WALLET_ADDRESS.lower():
        raise HTTPException(status_code=400, detail="Recipient does not match wallet address")
    if tx["value"] != expected_wei:
        raise HTTPException(status_code=400, detail=f"Amount mismatch: expected {expected_wei} wei, got {tx['value']}")


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
    if chain not in CHAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}")

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
        "billing": "monthly",
    }


@app.get("/v1/billing/crypto/verify")
def direct_evm_verify(tx_hash: str, request_id: str):
    if SUPABASE_URL:
        reused = _sb_get("payment_requests", {
            "tx_hash": f"eq.{tx_hash}",
            "status": "eq.paid",
        })
        if reused:
            raise HTTPException(status_code=400, detail="Transaction already used for a previous payment.")

        rows = _sb_get("payment_requests", {"request_id": f"eq.{request_id}"})
        if not rows:
            raise HTTPException(status_code=404, detail="Payment request not found")
        req = rows[0]
        if req.get("status") == "paid":
            raise HTTPException(status_code=400, detail="Payment already verified")
    else:
        for _rid, _req in _direct_evm_requests.items():
            if _req.get("tx_hash") == tx_hash and _req.get("status") == "paid":
                raise HTTPException(status_code=400, detail="Transaction already used for a previous payment.")

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
        _paid_keys[key] = {"expires_at": expires_at}

    return {
        "status": "paid",
        "api_key": key,
        "request_id": request_id,
        "chain": req["chain"],
        "amount_paid": float(req["amount_native"]),
        "expires_in_days": days,
        "usage": "Use X-API-Key header on /v1/ask-python and /v1/generate-script.",
    }


# ---------------------------------------------------------------- schemas
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


# ---------------------------------------------------------------- checkout UI
CHECKOUT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>Python Agent API Access</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#0b0f19; color:#e6e8ee; max-width:560px; margin:3rem auto; padding:0 1rem; }
    h1 { color:#7c8cf8; }
    .card { background:#141a2b; border:1px solid #26304a; border-radius:18px; padding:1.5rem; margin-bottom:1rem; }
    code { background:#0f1524; padding:0.35rem 0.6rem; border-radius:8px; word-break:break-all; }
    input { width:100%; padding:0.8rem; border-radius:10px; border:1px solid #2b3652; background:#0f1524; color:white; margin-bottom:1rem; }
    button { background:#7c8cf8; color:white; border:none; padding:0.8rem 1.2rem; border-radius:10px; font-weight:700; cursor:pointer; width:100%; }
    .result { margin-top:1rem; white-space:pre-wrap; color:#ffd166; }
  </style>
</head>
<body>
  <h1>🐍 Python Agent API Access</h1>
  <p>Pay <b>469 POL</b> to receive a paid API key valid for 30 days.</p>

  <div class="card">
    <p><b>Free Trial:</b> Get 50 requests for 7 days.</p>
    <button type="button" onclick="getFreeTrial()">Get Free Trial API Key</button>
    <div id="trial_result" class="result"></div>
  </div>

  <div class="card">
    <p><b>How to use your API key:</b></p>
    <p>Send requests with your key in the <code>X-API-Key</code> header.</p>
    <p>Example endpoint:</p>
    <code>POST https://python-agent-api.onrender.com/v1/ask-python</code>
  </div>

  <div class="card">
    <p><b>Chain:</b> __CHAIN__</p>
    <p><b>Pay exactly:</b><br><code>__AMOUNT__</code></p>
    <p><b>To wallet:</b><br><code>__WALLET__</code></p>
    <p><b>Request ID:</b> <span id="current_request_id">__REQUEST_ID__</span></p>
  </div>

  <div class="card">
    <p><b>Step 1:</b> Send exactly <b>469 POL</b> to the wallet above on Polygon.</p>
    <p><b>Step 2:</b> Paste your transaction hash below.</p>
    <input type="text" id="tx_hash" placeholder="0x...">
    <button type="button" onclick="verifyPayment()">Verify Payment & Get API Key</button>
    <div id="result" class="result"></div>
  </div>

  <script>
    function show(id, text) {
      document.getElementById(id).innerText = text;
    }

    function getFreeTrial() {
      show('trial_result', 'Requesting trial...');
      fetch('/v1/billing/free-trial', {method: 'POST'})
        .then(r => r.json())
        .then(data => {
          if (data.api_key) {
            show('trial_result', 'Trial key: ' + data.api_key + '\nRequests: ' + data.quota_limit + '\nExpires: ' + data.expires_in_days + ' days');
          } else {
            show('trial_result', JSON.stringify(data, null, 2));
          }
        })
        .catch(e => show('trial_result', 'Error: ' + e.message));
    }

    function verifyPayment() {
      var tx = document.getElementById('tx_hash').value.trim();
      var rid = document.getElementById('current_request_id').innerText;
      if (!tx) {
        show('result', 'Transaction hash missing.');
        return;
      }
      show('result', 'Verifying...');
      var url = '/v1/billing/crypto/verify?tx_hash=' + encodeURIComponent(tx) + '&request_id=' + encodeURIComponent(rid);
      fetch(url)
        .then(r => r.json())
        .then(data => {
          if (data.api_key) {
            show('result', 'Paid API key: ' + data.api_key);
          } else {
            show('result', JSON.stringify(data, null, 2));
          }
        })
        .catch(e => show('result', 'Error: ' + e.message));
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def checkout_page():
    try:
        payment = direct_evm_address(chain="polygon")
    except Exception:
        payment = {
            "request_id": "ERROR",
            "wallet_address": WALLET_ADDRESS,
            "chain": "polygon",
            "symbol": "POL",
            "amount": 469.0,
        }

    html = CHECKOUT_HTML
    html = html.replace("__CHAIN__", f"{payment['chain']} ({payment['symbol']})")
    html = html.replace("__AMOUNT__", f"{payment['amount']} {payment['symbol']}")
    html = html.replace("__WALLET__", payment["wallet_address"])
    html = html.replace("__REQUEST_ID__", payment["request_id"])
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/health")
def health():
    return {"status": "ok", "service": "neuro-symbolic-python-agent"}
