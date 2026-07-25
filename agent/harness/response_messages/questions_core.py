"""Shared pending-question rendering contracts."""


MESSAGES = {
    "question.instruction.options_or_value": {
        "en": "Reply with an option number or name, or enter a custom value.",
        "zh": "请回复选项编号或名称，也可以直接输入自定义值。",
        "arguments": {},
        "kinds": {"question_instruction"},
    },
    "question.instruction.value": {
        "en": "Enter a value.",
        "zh": "请直接输入值。",
        "arguments": {},
        "kinds": {"question_instruction"},
    },
    "question.instruction.option": {
        "en": "Reply with an option number or name.",
        "zh": "请回复选项编号或名称。",
        "arguments": {},
        "kinds": {"question_instruction"},
    },
}
