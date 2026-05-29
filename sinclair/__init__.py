"""
sinclair — a thin, opinionated agent layer over LangGraph.

Public API:
    Agent           — the motor; run() is stateless, chat() is stateful
    AgentConfig     — configuration dataclass
    AgentResult     — result of every run/send; same shape always
    Conversation    — stateful chat handle returned by Agent.chat()
    ConversationSnapshot — serializable snapshot for dump/load
    FinalAnswerError — raised when structured mode fails to converge
    PythonKernel    — stateful Python sandbox built in
    ExecutionResult — result returned by PythonKernel.execute()
    make_trace_observer — human-readable CLI trace observer
    ObserverFn      — Callable[[str, dict], None]
    log_observer    — structured logging observer
    print_observer  — stdout observer for local dev
"""

from ._agent import Agent, Conversation, ConversationSnapshot
from ._config import AgentConfig
from ._errors import FinalAnswerError
from ._metrics import AgentResult, IterMetrics
from ._obs import ObserverFn, log_observer, print_observer
from .python import ExecutionResult, PythonKernel
from .trace import make_trace_observer

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Conversation",
    "ConversationSnapshot",
    "ExecutionResult",
    "FinalAnswerError",
    "IterMetrics",
    "ObserverFn",
    "PythonKernel",
    "make_trace_observer",
    "log_observer",
    "print_observer",
]
