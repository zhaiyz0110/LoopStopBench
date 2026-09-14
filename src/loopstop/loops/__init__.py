from .base import LoopHarness, LoopTask, RoundContext  # noqa: F401
from .code_repair import CodeRepairLoop  # noqa: F401
from .writer_critic import WriterCriticLoop  # noqa: F401
from .react_qa import ReactQALoop  # noqa: F401

HARNESSES = {
    "code_repair": CodeRepairLoop,
    "writer_critic": WriterCriticLoop,
    "react_qa": ReactQALoop,
}
