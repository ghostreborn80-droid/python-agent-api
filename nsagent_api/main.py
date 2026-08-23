import os
import sys
import time
import json
import secrets
import requests
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/app")
sys.path.insert(0, "/content")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from web3 import Web3

from nsagent.agent import NeuroSymbolicAgent


WALLET_ADDRESS = os.environ.get(
    "WALLET_ADDRESS", "0x12133a4f996bdc9d2894b441e1f8621b499d2c3c"
)

CHAINS = {
    "polygon": {
        "rpcs": [
            "https://polygon.llamarpc.com",
            "https://polygon-rpc.com",
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
    "ethereum": {
        "rpcs": [
            "https://eth.llamarpc.com",
            "https://ethereum-rpc.publicnode.com",
        ],
        "chain_id": 1,
        "decimals": 18,
        "symbol": "ETH",
    },
    "bsc": {
        "rpcs": ["https://bsc-dataseed.binance.org"],
        "chain_id": 56,
        "decimals": 18,
        "symbol": "BNB",
    },
    "avalanche": {
        "rpcs": ["https://api.avax.network/ext/bc/C/rpc"],
        "chain_id": 43114,
        "decimals": 18,
        "symbol": "AVAX",
    },
    "arbitrum": {
        "rpcs": ["https://arb1.arbitrum.io/rpc"],
        "chain_id": 42161,
        "decimals": 18,
        "symbol": "ARB",
    },
    "optimism": {
        "rpcs": ["https://mainnet.optimism.io"],
        "chain_id": 10,
        "decimals": 18,
        "symbol": "OP",
    },
    "base": {
        "rpcs": ["https://mainnet.base.org"],
        "chain_id": 8453,
        "decimals": 18,
        "symbol": "ETH",
    },
}


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


_direct_evm_requests: Dict[str, Dict[str, Any]] = {}
_direct_evm_keys: Dict[str, str] = {}
_direct_evm_key_expiry: Dict[str, float] = {}
_trial_keys: Dict[str, Dict[str, Any]] = {}


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _sb_post(table: str, payload: dict):
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase insert error: {r.text[:200]}")


def _sb_get(table: str, filters: Optional[dict] = None) -> list:
    if not SUPABASE_URL:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, headers=_sb_headers(), params=filters or {}, timeout=30)
    if r.status_code >= 400:
        return []
    return r.json()


def _sb_patch(table: str, payload: dict, filters: dict):
    if not SUPABASE_URL:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.patch(url, headers=_sb_headers(), json=payload, params=filters, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase update error: {r.text[:200]}")


app = FastAPI(
    title="Neuro-Symbolic Python Agent API",
    version="3.0.0",
    description="Python code generation, knowledge, codebase intelligence, and crypto subscriptions.",
)


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


# ---------------------------------------------------------------- agent
_AGENT: Optional[NeuroSymbolicAgent] = None


def get_agent() -> NeuroSymbolicAgent:
    global _AGENT
    if _AGENT is None:
        project_root = os.environ.get("AGENT_PROJECT_ROOT", "/app/sample_project")
        state_path = os.environ.get("AGENT_STATE_PATH", "/app/agent_state/final_model_v3.json")
        _AGENT = NeuroSymbolicAgent(project_root=project_root, state_path=state_path)
    return _AGENT


# ---------------------------------------------------------------- api key auth
def require_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if x_api_key == "demo-key":
        return x_api_key

    paid_env = os.environ.get("PAID_API_KEYS", "")
    paid_keys = [k.strip() for k in paid_env.split(",") if k.strip()]
    if x_api_key in paid_keys:
        return x_api_key

    # Trial keys with quota.
    if SUPABASE_URL:
        trial_rows = _sb_get("trial_keys", {
            "api_key": f"eq.{x_api_key}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if trial_rows:
            row = trial_rows[0]
            used = int(row.get("quota_used", 0))
            limit = int(row.get("quota_limit", 50))
            if used >= limit:
                raise HTTPException(status_code=429, detail="Free trial quota exhausted. Upgrade to paid plan.")
            _sb_patch("trial_keys", {"quota_used": used + 1}, {"api_key": f"eq.{x_api_key}"})
            return x_api_key

        expired_trial = _sb_get("trial_keys", {"api_key": f"eq.{x_api_key}"})
        if expired_trial:
            raise HTTPException(status_code=401, detail="Free trial expired. Upgrade to paid plan.")

    # In-memory trial fallback.
    if x_api_key in _trial_keys:
        trial = _trial_keys[x_api_key]
        if time.time() > trial["expires_at"]:
            raise HTTPException(status_code=401, detail="Free trial expired. Upgrade to paid plan.")
        if trial["quota_used"] >= trial["quota_limit"]:
            raise HTTPException(status_code=429, detail="Free trial quota exhausted. Upgrade to paid plan.")
        trial["quota_used"] += 1
        return x_api_key

    # Paid keys from Supabase.
    if SUPABASE_URL:
        paid_rows = _sb_get("paid_keys", {
            "api_key": f"eq.{x_api_key}",
            "expires_at": f"gt.{int(time.time())}",
        })
        if paid_rows:
            return x_api_key

        any_paid = _sb_get("paid_keys", {"api_key": f"eq.{x_api_key}"})
        if any_paid:
            raise HTTPException(status_code=401, detail="API key expired. Pay for another month.")

    # In-memory paid fallback.
    if x_api_key in _direct_evm_keys:
        expiry = _direct_evm_key_expiry.get(x_api_key)
        if expiry and time.time() > expiry:
            raise HTTPException(status_code=401, detail="API key expired. Pay for another month.")
        return x_api_key

    raise HTTPException(status_code=401, detail="Invalid API key.")


# ---------------------------------------------------------------- billing
def _get_working_web3(chain: str) -> Web3:
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


def _verify_evm_tx(chain: str, tx_hash: str, expected_wei: int) -> None:
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
def create_free_trial():
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
    }


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


# ---------------------------------------------------------------- agent endpoints
def _extract_stdout(result: Any) -> Optional[str]:
    try:
        inner = getattr(result, "result", None)
        if inner is None:
            return None
        if isinstance(inner, dict):
            exec_res = inner.get("exec_result")
            if exec_res is not None and hasattr(exec_res, "stdout"):
                return (exec_res.stdout or "").strip()[:2000] or None
            if inner.get("generated"):
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
            message = getattr(tool_res, "message", "") or str(tool_res)
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
        return AgentResponse(request_type=request_type, message=message, trace=trace, status="ok")

    if rtype == "composite":
        outputs = result.get("outputs", {})
        return AgentResponse(
            request_type=request_type,
            message=f"Executed {len(outputs)} subgoals successfully.",
            trace=trace,
            status="ok",
        )

    if rtype == "refusal":
        return AgentResponse(
            request_type=request_type,
            message=message or "Request refused.",
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

    return AgentResponse(request_type=request_type, message=message or str(result), trace=trace, status="ok")


@app.post("/v1/ask-python", response_model=AgentResponse)
def ask_python(req: AskPythonRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    agent = get_agent()
    result = agent.handle(req.question)
    return _to_response(result, "ask_python")


@app.post("/v1/generate-script", response_model=AgentResponse)
def generate_script(req: GenerateScriptRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    agent = get_agent()
    result = agent.handle(req.task)
    return _to_response(result, "generate_script")


@app.post("/v1/codebase-query", response_model=AgentResponse)
def codebase_query(req: CodebaseQueryRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    require_api_key(x_api_key)
    agent = get_agent()
    result = agent.handle(req.question)
    return _to_response(result, "codebase_query")


# ---------------------------------------------------------------- checkout UI
CHECKOUT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Python Agent API Access</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#0b0f19; color:#e6e8ee; max-width:560px; margin:3rem auto; padding:0 1rem; }
    h1 { color:#7c8cf8; }
    .card { background:#141a2b; border:1px solid #26304a; border-radius:18px; padding:1.5rem; margin-bottom:1rem; }
    code { background:#0f1524; padding:0.35rem 0.6rem; border-radius:8px; word-break:break-all; }
    input { width:100%; padding:0.8rem; border-radius:10px; border:1px solid #2b3652; background:#0f1524; color:white; margin-bottom:1rem; }
    button { background:#7c8cf8; color:white; border:none; padding:0.8rem 1.2rem; border-radius:10px; font-weight:700; cursor:pointer; width:100%; }
    #result, #trial_result { margin-top:1rem; white-space:pre-wrap; }
  </style>
</head>
<body id="checkout-page">
  <h1>🐍 Python Agent API Access</h1>
  <p>Pay <b>469 POL</b> to receive a paid API key valid for 30 days.</p>

  <div class="card">
    <p><b>Free Trial:</b> Get 50 requests for 7 days.</p>
    <button onclick="getFreeTrial()">Get Free Trial API Key</button>
    <div id="trial_result"></div>
  </div>

  <div class="card">
    <p><b>Chain:</b> <span id="chain">LOADING</span></p>
    <p><b>Pay exactly:</b><br><code id="amount">LOADING</code></p>
    <p><b>To wallet:</b><br><code id="wallet">LOADING</code></p>
    <p><b>Request ID:</b> <span id="request_id">LOADING</span></p>
  </div>

  <div class="card">
    <p><b>Step 1:</b> Send exactly <b>469 POL</b> to the wallet above on Polygon.</p>
    <p><b>Step 2:</b> Paste your transaction hash below.</p>
    <input type="text" id="tx_hash" placeholder="0x..." />
    <button onclick="verifyPayment()">Verify Payment & Get API Key</button>
    <div id="result"></div>
  </div>

  <script>
    const INITIAL_PAYMENT = __PAYMENT_JSON__;
    let currentRequestId = INITIAL_PAYMENT.request_id;

    document.getElementById('chain').innerText = INITIAL_PAYMENT.chain + ' (' + INITIAL_PAYMENT.symbol + ')';
    document.getElementById('amount').innerText = INITIAL_PAYMENT.amount + ' ' + INITIAL_PAYMENT.symbol;
    document.getElementById('wallet').innerText = INITIAL_PAYMENT.wallet_address;
    document.getElementById('request_id').innerText = INITIAL_PAYMENT.request_id;

    async function getFreeTrial() {
      const resultDiv = document.getElementById('trial_result');
      resultDiv.innerText = 'Requesting trial...';
      const resp = await fetch('/v1/billing/free-trial', {method: 'POST'});
      const data = await resp.json();
      if (data.api_key) {
        resultDiv.innerText = 'Trial key: ' + data.api_key + '\\nRequests: ' + data.quota_limit + '\\nExpires: ' + data.expires_in_days + ' days';
      } else {
        resultDiv.innerText = JSON.stringify(data, null, 2);
      }
    }

    async function verifyPayment() {
      const tx = document.getElementById('tx_hash').value.trim();
      const resultDiv = document.getElementById('result');
      if (!tx || !currentRequestId) {
        resultDiv.innerText = 'Transaction hash missing.';
        return;
      }
      resultDiv.innerText = 'Verifying...';
      const resp = await fetch('/v1/billing/crypto/verify?tx_hash=' + encodeURIComponent(tx) + '&request_id=' + encodeURIComponent(currentRequestId));
      const data = await resp.json();
      if (data.api_key) {
        resultDiv.innerText = '✅ Payment verified. Your PAID API key is:\\n\\n' + data.api_key;
      } else {
        resultDiv.innerText = JSON.stringify(data, null, 2);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def checkout_page():
    try:
        payment = direct_evm_address(chain="polygon")
        payment_json = json.dumps(payment)
    except Exception:
        payment = {
            "request_id": "ERROR",
            "payment_id": 0,
            "wallet_address": WALLET_ADDRESS,
            "chain": "polygon",
            "symbol": "POL",
            "amount": 469.0,
            "amount_wei": 0,
        }
        payment_json = json.dumps(payment)

    html = CHECKOUT_HTML.replace("__PAYMENT_JSON__", payment_json)
    html = html.replace('<span id="chain">LOADING</span>', f'<span id="chain">{payment["chain"]} ({payment["symbol"]})</span>')
    html = html.replace('<code id="amount">LOADING</code>', f'<code id="amount">{payment["amount"]} {payment["symbol"]}</code>')
    html = html.replace('<code id="wallet">LOADING</code>', f'<code id="wallet">{payment["wallet_address"]}</code>')
    html = html.replace('<span id="request_id">LOADING</span>', f'<span id="request_id">{payment["request_id"]}</span>')
    return html


@app.get("/health")
def health():
    return {"status": "ok", "service": "neuro-symbolic-python-agent"}
