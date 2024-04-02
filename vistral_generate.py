import argparse
import asyncio
import json
import logging
import re

from tqdm import tqdm

from conversation import ConversationBatchService
from conversation.huggingface import HuggingfaceBatchService
from conversation.llama_cpp import LLaMACppBatchService
from conversation.openai import OpenAIBatchService
from enum_definitions import *
from data_definitions import *
from typing import Iterator
import yaml

import transformers
import torch

from dataset import mcocr
from sample_generation import *

# python generate.py COCO2014 --model_source openai --model gpt-3.5-turbo
def main():
    parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")

    # Optional parameters
    parser.add_argument("--dataset_storage_path", default="./filtered_data", type=str,
                        help="Storage path for downloaded dataset files")

    parser.add_argument("--output_path", default="instruct.jsonl", type=str,
                        help="Output file path for the created dataset (default: llava_instruct.json)")

    parser.add_argument("--output_format", default=OutputFormat.JSONL, type=OutputFormat,
                        choices=list(OutputFormat),
                        help="Output format of the dataset (default: llava-instruct). Available: " + ", ".join(
                            [o.value for o in OutputFormat]))

    # parser.add_argument("--context", default=[ContextProperties.CAPTIONS, ContextProperties.BOXES], nargs='*',
    #                     type=ContextProperties,
    #                     choices=list(ContextProperties),
    #                     help="Context appended to question and response generation queries (default: captions, boxes). Available: " + ", ".join(
    #                         [c.value for c in ContextProperties]))
    #
    # parser.add_argument("--question_context", default=[], nargs='*', type=QuestionContext,
    #                     choices=list(QuestionContext),
    #                     help="Context appended to output questions, visible at test time (default: none). Available: " + ", ".join(
    #                         [qc.value for qc in QuestionContext]))

    parser.add_argument("--model_source", default="huggingface",
                        type=ModelSourceType,
                        choices=list(ModelSourceType),
                        help="if huggingface: model must support conversational interface, "
                             "see https://huggingface.co/models?filter=conversational")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", type=str,
                        help="LLM to use for dataset generation.")

    parser.add_argument("--openai_base_url", default=None, type=str,
                        help="Base url of OpenAI compatible endpoint. Default is official OpenAI API endpoint.")

    parser.add_argument("--prompt_config", default="prompt_config_llava.yaml", type=str,
                        help="Prompt config")

    args = parser.parse_args()

    # Placeholder for the dataset generation logic.
    print("Generating dataset with the following parameters:")
    print("Sources: MC-OCR")
    print("Output Path: ", args.output_path)
    print("Output Format: ", args.output_format)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(args))


async def run(args):
    # model inference
    conversation_service: ConversationBatchService
    if args.model_source == ModelSourceType.OPENAI:
        conversation_service = OpenAIBatchService(args.model, args.openai_base_url)
    elif args.model_source == ModelSourceType.LLAMA_CPP:
        conversation_service = LLaMACppBatchService(args.model)
    elif args.model_source == ModelSourceType.HUGGINGFACE:
        conversation_service = HuggingfaceBatchService(args.model)
    else:
        raise ValueError



    pipe_name = re.sub(r'\W+', '_', args.model).lower()

    with open(args.prompt_config, 'r') as file:
        prompt_configs = yaml.safe_load(file)

    output_file = open(args.output_path, 'w')

    def on_result(question_id: str, result: str, context: Context, prompt_config):
        samples = process_llm_result(question_id, result, context, prompt_config)
        temp_list = []
        img_id = context.sample_id
        for sample in samples:
            sample_dict = sample.__dict__
            if len(temp_list) == 0:
                temp_list.append({"from": "human", "value": "<image>\n" + sample_dict["instruction"]})
                temp_list.append({"from": "gpt", "response": sample_dict["response"]})
            else:
                temp_list.append({"from": "human", "value": sample_dict["instruction"]})
                temp_list.append({"from": "gpt", "response": sample_dict["response"]})
        if temp_list != []:
            output_file.write(json.dumps({"id": img_id, "conversation": temp_list}, ensure_ascii=False) + "\n")

    conversation_service.set_on_result(on_result)

    # process datasets
        # data loader
    generator: Iterator[Context]
    generator = mcocr.MCOCRLoader(args.dataset_storage_path)
    num_expected_requests = len(generator) * len(prompt_configs.values())
    with tqdm(total=num_expected_requests, desc="Generating samples") as progress_bar:
        def on_result_progress(*vargs, **kwargs):
            progress_bar.update(1)
            progress_bar.set_description(f"{conversation_service.num_in_progress} running, {conversation_service.num_temp_failed} temp errors, {conversation_service.num_failed} failed (completed from cache {conversation_service.num_completed_from_cache})")
            progress_bar.refresh()
            on_result(*vargs, **kwargs)

        conversation_service.set_on_result(on_result_progress)

        tasks: List[asyncio.Task] = []
        for i, context in enumerate(generator):
            for prompt_config in prompt_configs.values():
                messages = generate_samples(context, prompt_config)

                # run pipeline and cache
                messages_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode('utf-8')).hexdigest()
                question_id = f"{pipe_name}_{generator.name}_{messages_hash}_{prompt_config['type']}"

                task = await conversation_service.submit(question_id, messages, context=context, prompt_config=prompt_config)
                tasks.append(task)

            if len(tasks) > 750:
                logging.info(f"Gathering {len(tasks)} tasks.")
                await asyncio.gather(*tasks)
                tasks.clear()

        await asyncio.gather(*tasks)
        await conversation_service.finish()


    output_file.close()


if __name__ == "__main__":
    main()
