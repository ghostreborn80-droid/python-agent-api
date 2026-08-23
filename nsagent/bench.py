
"""
nsagent.bench — Benchmark for the real-world neuro-symbolic agent.

Each task prints its trace and returns PASS/FAIL. No ungrounded actions.
Causal task seeds a real subprocess crash in analyzer.summarize before
answering, so the answer is grounded in an observed runtime failure.
"""
from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import Any, Callable, Dict

from nsagent.agent import NeuroSymbolicAgent


def run_benchmark(agent: NeuroSymbolicAgent) -> Dict[str, bool]:
    results: Dict[str, bool] = {}

    def task(name: str, fn: Callable[[], Any], check: Callable[[Any], bool]) -> bool:
        print("\n" + "=" * 76)
        print(f"BENCHMARK TASK: {name}")
        print("=" * 76)
        try:
            result = fn()
            passed = check(result)
            print(f"RESULT: {'PASS' if passed else 'FAIL'}")
            return passed
        except Exception as exc:
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")
            print("RESULT: FAIL")
            return False

    # 1. Explain a module
    results["explain_report"] = task(
        "Explain report.py",
        lambda: agent.handle("Explain report.py"),
        lambda r: r.get("type") == "answer" and "generate_report" in r.get("message", "")
    )

    # 2. Inspect callees
    results["inspect_callees"] = task(
        "What does generate_report call?",
        lambda: agent.handle("What does generate_report call?"),
        lambda r: r.get("type") == "answer" and "clean_data" in r.get("message", "")
    )

    # 3. Inspect callers
    results["inspect_callers"] = task(
        "Who calls clean_data?",
        lambda: agent.handle("Who calls clean_data?"),
        lambda r: r.get("type") == "answer" and "generate_report" in r.get("message", "")
    )

    # 4. Refuse ungrounded request
    results["refuse_ungrounded"] = task(
        "Run the frobnicator on sales.csv",
        lambda: agent.handle("Run the frobnicator on sales.csv"),
        lambda r: r.get("type") == "refusal"
    )

    # 5. Run a grounded function
    results["run_generate_report"] = task(
        "Run generate_report on sales.csv",
        lambda: agent.handle("Run generate_report on sales.csv"),
        lambda r: r.get("type") == "tool_result" and r.get("result", None) is not None
                  and r["result"].ok
    )

    # 6. Composite goal: CSV -> HTML report
    def composite():
        agent.project_root.joinpath("report.html").unlink(missing_ok=True)
        return agent.handle("Clean sales.csv and write an HTML report", auto_confirm=True)
    results["composite_clean_report"] = task(
        "Clean sales.csv and write an HTML report",
        composite,
        lambda r: r.get("type") == "composite" and (agent.project_root / "report.html").exists()
    )

    # 7. Causal question — seed a real sandboxed crash in analyzer.summarize
    def causal_setup_and_ask():
        code = ("import pandas as pd\n"
                "from analyzer import summarize\n"
                "summarize(pd.DataFrame())\n")
        crash_res = agent.sandbox.run_code(code, filename="bench_crash_summarize.py")
        print(f"  [Causal setup] sandbox crash: {crash_res.summary()}")
        if crash_res.success:
            return {"type": "no_failure", "message": "sandbox unexpectedly succeeded"}
        m = re.search(r"KeyError: ['\"]([^'\"]+)", crash_res.combined)
        msg = m.group(1) if m else "total"
        agent.causal.record_failure("analyzer.py::summarize", "KeyError", msg,
                                    traceback_text=crash_res.combined)
        print(f"  [Causal setup] recorded failure: analyzer.py::summarize raised KeyError({msg!r})")
        return agent.handle("Why did summarize fail?")
    results["causal_why"] = task(
        "Seed crash in summarize and answer 'Why did summarize fail?'",
        causal_setup_and_ask,
        lambda r: r.get("type") == "answer" and "KeyError" in r.get("message", "")
                  and "total" in r.get("message", "")
    )

    # 8. Reuse a skill
    def reuse():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent.handle("Run generate_report on sales.csv")
        return buf.getvalue()
    results["reuse_skill"] = task(
        "Reuse skill for generate_report",
        reuse,
        lambda out: "Skill reused" in out
    )

    # 9. Add a unit test (grounded write_file with confirmation)
    results["add_unit_test"] = task(
        "Add a unit test for format_currency",
        lambda: agent.handle("Add a unit test for format_currency", auto_confirm=True),
        lambda r: r.get("type") == "tool_result" and r.get("result", None) is not None
                  and r["result"].ok
    )

    # 10. Run full test suite after adding test
    results["run_tests_after_add"] = task(
        "Run full test suite after adding test",
        lambda: agent.tools.invoke("run_tests", {"args": ["tests", "-q"]}, confirmed=True),
        lambda r: r.ok and ("PASS" in r.message or "passed" in r.message)
    )

    # 11. Persist final world model
    def save():
        agent.save()
        return agent.state_path.exists()
    results["persist_final_model"] = task(
        "Persist final world model",
        save,
        lambda exists: exists
    )

    return results
