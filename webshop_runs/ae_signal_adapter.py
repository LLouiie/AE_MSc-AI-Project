"""WebShop operationalisation of observable AE trajectory signals."""
from __future__ import annotations
import re
from ae.controllers.signals import SignalExtractor, jaccard, normalize_text

_ACTION = re.compile(r"^(search|click|think)\[(.*)\]$", re.DOTALL)

class WebShopAdmissible(list):
    def __init__(self, commands=(), *, state_signature=()):
        super().__init__(commands)
        self.state_signature = tuple(state_signature)

def _action_signature(action: str):
    match = _ACTION.match(action.strip())
    if not match:
        return None
    return match.group(1), normalize_text(match.group(2))

class WebShopSignalExtractor(SignalExtractor):
    """Keep AE fields/ranges while measuring them from WebShop state."""
    def extract(self, **kwargs):
        signals = super().extract(**kwargs)
        action = kwargs["action"]
        observation = kwargs["observation"]
        recent_actions = kwargs["recent_actions"]
        recent_observations = kwargs["recent_observations"]
        before = kwargs["admissible_before"]
        after = kwargs.get("admissible_after")
        if after is None:
            after = before
        is_think = kwargs["is_think_action"]
        signature = _action_signature(action)
        window = self.config.repetition_window
        recent = recent_actions[-window:] if window > 0 else []
        signals.repeated_action = max(
            (1.0 if signature is not None and signature == _action_signature(old) else 0.0
             for old in recent), default=0.0,
        )
        previous = _action_signature(recent_actions[-1]) if recent_actions else None
        signals.consecutive_exact_action_repeat = (
            1.0 if signature is not None and signature == previous else 0.0
        )
        state_changed = (
            getattr(before, "state_signature", ())
            != getattr(after, "state_signature", ())
        )
        if is_think:
            # OK. is a protocol placeholder, not a repeated world observation.
            signals.repeated_observation = 0.0
            signals.observation_novelty = 0.0
            signals.local_state_change_proxy = 0.0
            signals.unexpected_outcome = 0.0
            signals.information_gain_proxy = 0.0
            signals.progress_signal = 0.0
            return signals
        recent_obs = recent_observations[-window:] if window > 0 else []
        signals.repeated_observation = max(
            (jaccard(observation, old) for old in recent_obs
             if normalize_text(old) != "ok."), default=0.0,
        )
        signals.observation_novelty = 1.0 - signals.repeated_observation
        valid = signals.invalid_action < 0.5
        signals.local_state_change_proxy = 1.0 if valid and state_changed else 0.0
        signals.unexpected_outcome = 1.0 if valid and not state_changed else 0.0
        signals.information_gain_proxy = 1.0 if (
            valid and (state_changed or
                       signals.repeated_observation < self.config.novelty_progress_threshold)
        ) else 0.0
        signals.progress_signal = (
            signals.observation_novelty if valid and state_changed else 0.0
        )
        return signals
