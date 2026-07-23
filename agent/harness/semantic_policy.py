"""Shared semantic policy fragments used by every intent authority."""

PENDING_CANDIDATE_SEMANTIC_POLICY = (
    "Syntax candidates are not selections. A value mentioned only as an example, "
    "quotation, rejected option, negated operation, correction target, or value "
    "the user explicitly says not to apply does not answer the pending question. "
    "Preserve any independent explanation or correction demand instead. "
    "A source may affirm one exact candidate while excluding a category of "
    "alternatives. Excluding documentation, sample, old, or unrelated alternatives "
    "does not negate the separately affirmed candidate. Preserve unambiguous "
    "operation framing and alternative exclusions as support for that candidate; "
    "a negation or example of the selected candidate remains non-selecting. "
    "When two or more values satisfy the active manual pending contract, never "
    "select one from list order. Exactly one candidate must be distinguished as "
    "selected and every other candidate must be distinguished as rejected, old, "
    "example-only, or otherwise not selected. A conjunction, disjunction, "
    "slash-separated list, comparison, or bare sequence is unresolved. "
)
