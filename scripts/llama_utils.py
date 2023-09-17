"""Helper functions for preparing content for Llama models. Much of this code
is modified from the official repos:
- https://github.com/facebookresearch/codellama/blob/main/llama/generation.py
"""
from typing import List, Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


def prepare_chats(dialogs: List[Dialog]) -> List[str]:
    texts = []
    for dialog in dialogs:
        if any(
            [tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog]):
            raise ValueError(
                f"Special tags {SPECIAL_TAGS} are not allowed as part of the prompt."
            )
        # absorb system prompt into first user prompt, if present
        if dialog[0]["role"] == "system":
            dialog = [{
                "role":
                dialog[1]["role"],
                "content":
                B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }] + dialog[2:]
        # check that dialog starts with user and alternates
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all([
            msg["role"] == "assistant" for msg in dialog[1::2]
        ]), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        # modified implementation below to prepare text, not tokenized inputs
        # NB: we rely on HF tokenizers to include <s> tag at the start of the text but
        # if we have multiple turns, we need to add <s> and </s> tags ourselves
        # between turns in the conversation
        # see more details here: https://huggingface.co/blog/codellama#conversational-instructions
        history = ""
        for prompt, answer in zip(dialog[::2], dialog[1::2]):
            history += f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}{BOS}"

        # add last user message, note that there's no trailing space
        assert (dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
        history += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

        texts.append(history)
    return texts