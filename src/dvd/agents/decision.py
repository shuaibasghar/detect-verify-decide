"""Decision Agent: fuses detector confidence c with the verifier verdict v.

    accept = (c >= tau_high) OR (v == yes AND c >= tau_low)

  * c >= tau_high  -> accepted without consulting the verifier (confident detections)
  * c <  tau_low   -> rejected without consulting the verifier
  * otherwise      -> the verifier's verdict decides (the "uncertain band")
Only the uncertain band costs verifier calls, which is what bounds the added latency.
"""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# re: regular expressions, used to read the LLM aggregator's one-word answer.
import re

# numpy: vectorised comparisons over all candidates of an image.
import numpy as np


def rule_accept(scores, verdicts, tau_low: float, tau_high: float) -> np.ndarray:
    """The decision rule. Returns a boolean array: True = keep the candidate."""
    c = np.asarray(scores, dtype=np.float32)
    v = np.asarray(verdicts, dtype=bool)
    # Confident detections pass alone; others need the verifier's "yes" AND c above the lower threshold.
    return (c >= tau_high) | (v & (c >= tau_low))


def needs_verification(scores, tau_low: float, tau_high: float) -> np.ndarray:
    """Which candidates fall in the uncertain band [tau_low, tau_high) and therefore need the verifier."""
    c = np.asarray(scores, dtype=np.float32)
    return (c >= tau_low) & (c < tau_high)


class RuleDecisionAgent:
    """The deployable Decision Agent: a thin object wrapper around the two functions above."""

    def __init__(self, tau_low: float, tau_high: float):
        # The band must be non-empty or inverted thresholds would make the rule meaningless.
        assert tau_low <= tau_high, "tau_low must be <= tau_high"
        self.tau_low, self.tau_high = tau_low, tau_high

    def needs_verification(self, scores) -> np.ndarray:
        """Gate: which candidates should be sent to the Verifier Agent."""
        return needs_verification(scores, self.tau_low, self.tau_high)

    def decide(self, scores, verdicts) -> np.ndarray:
        """Final accept/reject mask from detector scores and verifier verdicts."""
        return rule_accept(scores, verdicts, self.tau_low, self.tau_high)


# Prompt for the optional LLM aggregator: it sees the label, c, the verdict and P(yes), never the image.
AGG_PROMPT = (
    "An object detector proposed a '{label}' with confidence {c:.2f} (0 = unsure, 1 = certain). "
    "An independent verifier looked only at a zoomed-in crop of that region and answered '{v}' "
    "to the question whether the object is present (its probability of 'yes' was {p:.2f}). "
    "Should the detection be kept? Answer with exactly one word: accept or reject."
)


class LLMAggregatorAgent:
    """Optional alternative to the rule: a text-only LLM fuses the two signals.

    Uses the verifier's own language model in text-only mode, so no extra model is loaded.
    Every candidate costs one extra generation call, so this is evaluated on a small subset.
    """

    def __init__(self, verifier, max_new_tokens: int = 3):
        # Reuse the already-loaded VLM (text-only), and allow only a few output tokens ("accept"/"reject").
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def decide(self, labels, scores, verdicts, p_yes) -> np.ndarray:
        """Ask the LLM about each candidate; True where its answer contains 'accept'."""
        # Fill the prompt template once per candidate.
        prompts = [AGG_PROMPT.format(label=l, c=float(c), v="yes" if v else "no", p=float(p))
                   for l, c, v, p in zip(labels, scores, verdicts, p_yes)]
        if not prompts:
            return np.zeros(0, bool)
        # One batched text-only generation.
        outs = self.verifier.generate_text(prompts, max_new_tokens=self.max_new_tokens)
        # Word-boundary match so e.g. "unaccepted" would not count.
        return np.array([bool(re.search(r"\baccept", o.lower())) for o in outs])
