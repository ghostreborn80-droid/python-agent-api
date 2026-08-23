
"""
nsagent_api.evm_billing — Direct EVM billing with fallback RPCs.

Supports mainnet + testnets. Each chain has a list of RPC URLs;
the verifier tries them in order until one works.
"""
from __future__ import annotations

import os
import sys
import time
import json
import secrets
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/content")

from fastapi import FastAPI, Header, HTTPException
from web3 import Web3

WALLET_ADDRESS = os.environ.get(
    "WALLET_ADDRESS", "0x12133a4f996bdc9d2894b441e1f8621b499d2c3c"
)

# Each chain maps to a list of RPCs. The first working one wins.
CHAINS: Dict[str, Dict[str, Any]] = {
    # Mainnets
    "ethereum": {
        "rpc": [
            "https://eth.llamarpc.com",
            "https://ethereum-rpc.publicnode.com",
            "https://rpc.ankr.com/eth",
        ],
        "chain_id": 1, "decimals": 18, "symbol": "ETH"
    },
    "polygon": {
        "rpc": [
            "https://polygon.llamarpc.com",
            "https://polygon-rpc.com",
            "https://rpc.ankr.com/polygon",
        ],
        "chain_id": 137, "decimals": 18, "symbol": "MATIC"
    },
    "bsc": {
        "rpc": [
            "https://bsc-dataseed.binance.org",
            "https://bsc-dataseed1.binance.org",
            "https://rpc.ankr.com/bsc",
        ],
        "chain_id": 56, "decimals": 18, "symbol": "BNB"
    },
    "avalanche": {
        "rpc": [
            "https://api.avax.network/ext/bc/C/rpc",
            "https://avalanche-c-chain-rpc.publicnode.com",
        ],
        "chain_id": 43114, "decimals": 18, "symbol": "AVAX"
    },
    "arbitrum": {
        "rpc": [
            "https://arb1.arbitrum.io/rpc",
            "https://arbitrum-one-rpc.publicnode.com",
        ],
        "chain_id": 42161, "decimals": 18, "symbol": "ARB"
    },
    "optimism": {
        "rpc": [
            "https://mainnet.optimism.io",
            "https://optimism-rpc.publicnode.com",
        ],
        "chain_id": 10, "decimals": 18, "symbol": "OP"
    },
    "base": {
        "rpc": [
            "https://mainnet.base.org",
            "https://base-rpc.publicnode.com",
        ],
        "chain_id": 8453, "decimals": 18, "symbol": "ETH"
    },

    # Testnets — free tokens
    "polygon-amoy": {
        "rpc": [
            "https://rpc-amoy.polygon.technology",
            "https://polygon-amoy.blockpi.network/v1/rpc/public",
            "https://polygon-amoy.public.blastapi.io",
            "https://rpc.ankr.com/polygon_amoy",
            "https://polygon-amoy.g.alchemy.com/v2/demo",
        ],
        "chain_id": 80002, "decimals": 18, "symbol": "tMATIC"
    },
    "ethereum-sepolia": {
        "rpc": [
            "https://ethereum-sepolia-rpc.publicnode.com",
            "https://rpc.sepolia.org",
            "https://rpc2.sepolia.org",
            "https://1rpc.io/sepolia",
        ],
        "chain_id": 11155111, "decimals": 18, "symbol": "tETH"
    },
}

_direct_evm_requests: Dict[str, Dict[str, Any]] = {}
_direct_evm_keys: Dict[str, str] = {}

app = FastAPI(title="Direct EVM Billing Service (fallback RPC)", version="0.2.0")


def _make_payment_amount(payment_id: int, decimals: int) -> int:
    """Amount in wei: 0.001 native coin + payment id embedded after decimal."""
    base = 10 ** (decimals - 3)  # 0.001
    micro = payment_id % (10 ** 6)
    return base + micro * 10 ** (decimals - 12)


def _get_working_web3(chain: str) -> Web3:
    cfg = CHAINS.get(chain)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}")

    for rpc in cfg["rpc"]:
        w3 = Web3(Web3.HTTPProvider(rpc))
        if w3.is_connected():
            return w3

    raise HTTPException(status_code=502, detail=f"All RPCs for {chain} are currently unavailable.")


def _verify_evm_tx(chain: str, tx_hash: str, expected_wei: int) -> None:
    w3 = _get_working_web3(chain)
    try:
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Transaction not found: {exc}")

    if tx["to"] and tx["to"].lower() != WALLET_ADDRESS.lower():
        raise HTTPException(status_code=400, detail="Recipient does not match wallet address")
    if tx["value"] != expected_wei:
        raise HTTPException(status_code=400, detail=f"Amount mismatch: expected {expected_wei} wei, got {tx['value']}")



from fastapi.responses import HTMLResponse

CHECKOUT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Buy API Access</title>
  <style>
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0b0f19;
      color: #e6e8ee;
      max-width: 560px;
      margin: 3rem auto;
      padding: 0 1rem;
    }
    h1 { color: #7c8cf8; }
    .card {
      background: #141a2b;
      border: 1px solid #26304a;
      border-radius: 18px;
      padding: 1.5rem;
    }
    code {
      background: #0f1524;
      padding: 0.35rem 0.6rem;
      border-radius: 8px;
      word-break: break-all;
    }
    button {
      background: #7c8cf8;
      color: white;
      border: none;
      padding: 0.8rem 1.2rem;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
    }
    input {
      width: 100%;
      padding: 0.8rem;
      border-radius: 10px;
      border: 1px solid #2b3652;
      background: #0f1524;
      color: white;
      margin-bottom: 1rem;
    }
    #result { margin-top: 1rem; white-space: pre-wrap; }
  </style>
</head>
<body id="checkout-page">
  <h1>🐍 Python Agent API Access</h1>
  <p>Pay in crypto to receive a paid API key.</p>

  <div class="card">
    <p><b>Chain:</b> <span id="chain">...</span></p>
    <p><b>Pay exactly:</b><br><code id="amount">...</code></p>
    <p><b>To wallet:</b><br><code id="wallet">...</code></p>
    <p><b>Request ID:</b> <span id="request_id"></span></p>
  </div>

  <div class="card" style="margin-top:1.2rem">
    <p><b>Step 1:</b> Send the exact amount above.</p>
    <p><b>Step 2:</b> Paste your transaction hash below.</p>
    <input type="text" id="tx_hash" placeholder="0x..." />
    <button onclick="verifyPayment()">Verify Payment & Get API Key</button>
    <div id="result"></div>
  </div>

  <script>
    let currentRequestId = null;

    async function loadPayment() {
      const resp = await fetch('/v1/billing/crypto/direct-address?chain=polygon-amoy');
      const data = await resp.json();
      currentRequestId = data.request_id;
      document.getElementById('chain').innerText = data.chain + ' (' + data.symbol + ')';
      document.getElementById('amount').innerText = data.amount + ' ' + data.symbol;
      document.getElementById('wallet').innerText = data.wallet_address;
      document.getElementById('request_id').innerText = data.request_id;
    }

    async function verifyPayment() {
      const tx = document.getElementById('tx_hash').value.trim();
      const resultDiv = document.getElementById('result');
      if (!tx || !currentRequestId) {
        resultDiv.innerText = 'Transaction hash missing or payment request not loaded.';
        return;
      }
      resultDiv.innerText = 'Verifying...';
      const resp = await fetch(
        '/v1/billing/crypto/verify?tx_hash=' + encodeURIComponent(tx) +
        '&request_id=' + encodeURIComponent(currentRequestId)
      );
      const data = await resp.json();
      if (data.api_key) {
        resultDiv.innerText = '✅ Payment verified. Your PAID API key is:\\n\\n' + data.api_key;
      } else {
        resultDiv.innerText = JSON.stringify(data, null, 2);
      }
    }

    loadPayment();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def checkout_page():
    return CHECKOUT_HTML


@app.get("/health")
def health():
    return {"status": "ok", "service": "direct-evm-billing", "wallet": WALLET_ADDRESS}


@app.get("/v1/billing/crypto/direct-address")
def direct_evm_address(chain: str = "polygon-amoy"):
    if chain not in CHAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {chain}. Available: {list(CHAINS)}")

    payment_id = secrets.randbelow(900000) + 100000
    decimals = CHAINS[chain]["decimals"]
    amount_wei = _make_payment_amount(payment_id, decimals)
    amount_native = amount_wei / (10 ** decimals)

    request_id = secrets.token_hex(8)
    _direct_evm_requests[request_id] = {
        "payment_id": payment_id,
        "chain": chain,
        "amount_wei": amount_wei,
        "amount_native": amount_native,
        "wallet": WALLET_ADDRESS,
        "status": "waiting",
        "created_at": time.time(),
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
    }


@app.get("/v1/billing/crypto/verify")
def direct_evm_verify(tx_hash: str, request_id: str):
    if request_id not in _direct_evm_requests:
        raise HTTPException(status_code=404, detail="Payment request not found")

    req = _direct_evm_requests[request_id]
    if req["status"] == "paid":
        raise HTTPException(status_code=400, detail="Payment already verified")

    _verify_evm_tx(req["chain"], tx_hash, req["amount_wei"])

    req["status"] = "paid"
    key = "sk_live_" + secrets.token_urlsafe(24)
    _direct_evm_keys[key] = "evm-paid-user"

    return {
        "status": "paid",
        "api_key": key,
        "request_id": request_id,
        "chain": req["chain"],
        "amount_paid": req["amount_native"],
    }
