class Scheduler:
    """决定每题结束后是否触发巩固。所有方法共用此接口。"""
    def decide(self, state: dict) -> str:
        """state 字段:
        q_index      当前题号(从 1 起)
        em           本题最终是否答对
        trials_used  本题用了几轮
        # Phase 2 预留: confidence, disagreement, frustration
        返回 "consolidate" 或 "continue"
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class NeverScheduler(Scheduler):
    """No Reflection / Reflexion 基线用:永不巩固。"""
    def decide(self, state):
        return "continue"


class FixedScheduler(Scheduler):
    """每 k 题巩固一次。k=None 表示 end-only(k=∞, ExpeL 本尊),
    由 runner 在练习循环结束后统一调一次 consolidate。"""
    def __init__(self, k=None):
        self.k = k

    def decide(self, state):
        if self.k is None:
            return "continue"          # end-only: 循环内永不触发
        return "consolidate" if state["q_index"] % self.k == 0 else "continue"

    def name(self):
        return f"Fixed(k={self.k if self.k else 'end'})"


def build_scheduler(spec: str) -> Scheduler:
    """命令行解析: never | fixed:10 | fixed:end | (Phase 2: affect)"""
    if spec == "never":
        return NeverScheduler()
    if spec.startswith("fixed:"):
        v = spec.split(":", 1)[1]
        return FixedScheduler(None if v == "end" else int(v))
    raise ValueError(f"未知 scheduler: {spec}")