import hashlib
import json
from textwrap import dedent
import random
from typing import Callable, Dict

from data_definitions import *


def generate_samples(context: Context, prompt_config):
    include_captions = "captions" in prompt_config["inputs"]
    captions_str = "\n".join(x for x in context.captions[0].caption) if include_captions else ""

    instruction = captions_str

    # construct prompt
    messages = [
        {"role": "system", "content": prompt_config["system_prompt"]}
    ]
    for sample in prompt_config["examples"]:
        messages.append({"role": "user", "content": sample["input"]})
        messages.append({"role": "assistant", "content": sample["output"]})
    messages.append({"role": "user", "content": instruction})

    return messages

def process_llm_result(question_id: str, result: str, context: Context, prompt_config):
    samples: List[Sample] = []

    if isinstance(result, dict):
        print(f"{question_id} Error result: {result}")
        return []

    type = prompt_config["type"]

    def remove_stopwords_strip(text: str) -> str:
        if "stopwords" in prompt_config:
            for sw in prompt_config["stopwords"]:
                text = text.replace(sw, "")
        return text.strip()

    if "split_user_assistant" in prompt_config and prompt_config["split_user_assistant"]:
        result = result.split(prompt_config["split_user_assistant"])
        if not len(result) % 2 == 0:
            print(f"Error {type}: Expecting on assistant answer for every user question. Got {len(result)} messages. \n result: {result}")
            return []
        for i in range(0, len(result), 2):
            samples.append(Sample(
                id=question_id,
                instruction=remove_stopwords_strip(result[i]),
                response=remove_stopwords_strip(result[i + 1]),
                image=context.sample_id,
                image_source=context.source,
                type=type
            ))
    else:
        samples.append(Sample(
            id=question_id,
            instruction=random.choice(prompt_config["instructions"]),
            response=remove_stopwords_strip(result),
            image=context.sample_id,
            image_source=context.source,
            type=type
        ))

    return samples