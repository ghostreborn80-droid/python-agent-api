"""nsagent — neuro-symbolic software-engineering agent.

Modules (built incrementally, one block at a time):
  memory    — RealCodeMemory: symbolic world model        (this block, R2)
  sandbox   — isolated subprocess execution               (next, R1)
  compiler  — NL -> PlanSpec with graph grounding         (R3)
  tools     — typed tool registry + confirmation gates    (R4)
  planner   — composite-goal DAG decomposition            (R6)
  healer    — shadow-patch self-healing loop              (R3)
  agent     — orchestrator + episodic memory              (R5)
  bench     — >=10-task benchmark with traces             (R8)
"""
__version__ = "0.1.0"
