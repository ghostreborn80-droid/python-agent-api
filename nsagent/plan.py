"""
nsagent.plan — PlanSpec: a typed, inspectable plan compiled from NL.

A PlanSpec is GROUNDED only when every entity it references resolves to a
node in the RealCodeMemory graph. Ungrounded plans are REFUSED with a
clarification question, per the architectural invariant:
"Neural proposes, graph disposes."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanSpec:
    request: str
    intent: str                       # execute | explain | inspect_callees | inspect_callers | unknown
    target: Optional[str] = None      # qualified node id, e.g. report.py::generate_report
    target_name: Optional[str] = None # bare name for display
    args: List[str] = field(default_factory=list)
    grounded: bool = False
    note: str = ""
    clarification: Optional[str] = None
    tool: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "GROUNDED" if self.grounded else "UNGROUNDED"
        s = (f"PlanSpec(intent={self.intent}, target={self.target or '?'}, "
             f"target_name={self.target_name or '?'}, args={self.args}, {status})")
        if self.note:
            s += f"\n  note: {self.note}"
        if self.clarification:
            s += f"\n  clarification: {self.clarification}"
        if self.tool:
            s += f"\n  tool: {self.tool}{self.tool_params}"
        return s

    def to_tool_call(self) -> Optional[tuple]:
        """Map this grounded plan to a tool name + params, or None."""
        if not self.grounded:
            return None
        if self.intent == "execute":
            return ("run_function", {"function": self.target, "args": self.args})
        if self.intent == "inspect_callees":
            return ("query_callees", {"function": self.target})
        if self.intent == "inspect_callers":
            return ("query_callers", {"function": self.target})
        if self.intent == "generate_python_script":
            return ("generate_python_script", {"task": self.request})
        if self.intent == "python_knowledge":
            return ("python_knowledge", {"question": self.request})
        return None
