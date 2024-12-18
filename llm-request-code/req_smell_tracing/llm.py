import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import openai
import requests
from openai import OpenAI

from req_smell_tracing.requirements.requirement import Requirement

logger = logging.getLogger(__name__)


class Prompt:
    def __init__(
        self,
        config: dict[str, str | None],
        role: str,
        content: str | None = None,
        content_file: str | None = None,
    ):
        self.role = role

        if content and content_file:
            raise ValueError("Content and content_file cannot be provided together")

        if not content and not content_file:
            raise ValueError("Either content or content_file must be provided")

        if content:
            self.content = content
        elif content_file:
            path = os.path.join(config["DATA_PATH"], "prompts", content_file)

            try:
                with open(path, "r") as f:
                    self.content = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"Content file not found at {path}")

    def fill_content(self, code, requirements: list[Requirement]):
        self.content = re.sub("{{code}}", Prompt.code_numbered(code), self.content)
        self.content = re.sub(
            "{{requirements}}", Requirement.format_as_list(requirements), self.content
        )

    @staticmethod
    def code_numbered(code: str) -> str:
        return "\n".join(
            map(lambda x: f"{x[0] + 1}. {x[1]}", enumerate(code.splitlines()))
        )


class LLM(ABC):
    @abstractmethod
    def __init__(
        self,
        config: dict[str, str | None],
        model: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "json",
    ): ...

    @abstractmethod
    def prompt(
        self, result_id: str, prompts: list[Prompt], add_to_batch: bool = False
    ) -> Optional[str]: ...

    @staticmethod
    @abstractmethod
    def run_batch(path: str, dry_run: bool = False) -> Optional[str]: ...


class GPT(LLM):
    def __init__(
        self,
        config: dict[str, str | None],
        model: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "json",
    ):
        self.client = OpenAI(api_key=config["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.prompts_path = os.path.join(config["DATA_PATH"], "prompts")
        self.chat_completions = []

    def prompt(
        self, result_id: str, prompts: list[Prompt], add_to_batch: bool = False
    ) -> Optional[str]:
        messages = []

        for prompt in prompts:
            messages.append({"role": prompt.role, "content": prompt.content})

        logger.info(f"Prompting GPT model {self.model}")

        if self.response_format == "json":
            response_format = {"type": "json_object"}
        else:
            response_format = None

        if add_to_batch:
            completion = {
                "custom_id": result_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "response_format": response_format,
                    "messages": messages,
                },
            }

            logger.info("Writing completion to batch file")

            with open("batch.jsonl", "a") as f:
                f.write(f"{json.dumps(completion)}\n")

            return None
        else:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                    messages=messages,
                )
            except openai.error.Timeout as e:
                logger.error(f"Timeout while prompting GPT model: {e}")
                return None
            except openai.error.APIError as e:
                logger.error(f"API error while prompting GPT model: {e}")
                return None
            except openai.error.APIConnectionError as e:
                logger.error(f"API connection error while prompting GPT model: {e}")
                return None
            except openai.error.InvalidRequestError as e:
                logger.error(f"Invalid request error while prompting GPT model: {e}")
                return None
            except openai.error.AuthenticationError as e:
                logger.error(f"Authentication error while prompting GPT model: {e}")
                return None
            except openai.error.PermissionError as e:
                logger.error(f"Permission error while prompting GPT model: {e}")
                return None
            except openai.error.RateLimitError as e:
                logger.error(f"Rate limit error while prompting GPT model: {e}")
                return None

            if len(chat_completion.choices) > 0:
                res = chat_completion.choices[0]

                match res.finish_reason:
                    case "length":
                        logger.error("Chat completion finished due to length")
                        return None
                    case "content_filter":
                        logger.error("Chat completion finished due to content filter")
                        return None
                    case "stop":
                        logger.info("Chat completion finished due to stop")
                    case _:
                        logger.error("Chat completion finished due to unknown reason")
                        return None

                return res.message.content

            logger.error("Chat completion finished without any choices")
            return None

    @staticmethod
    def run_batch(config, dry_run: bool = False) -> Optional[str]:
        client = OpenAI(api_key=config["OPENAI_API_KEY"])

        if dry_run:
            logger.info("Dry run completed")
            return None
        else:
            try:
                batch_input_file = client.files.create(
                    file=open("batch.jsonl", "rb"), purpose="batch"
                )
                batch_input_file_id = batch_input_file.id

                batch = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": "all experiments",
                    },
                )

                logger.info("Batch started")

                return batch.id
            except openai.error.Timeout as e:
                logger.error(f"Timeout while prompting GPT model: {e}")
                return None
            except openai.error.APIError as e:
                logger.error(f"API error while prompting GPT model: {e}")
                return None
            except openai.error.APIConnectionError as e:
                logger.error(f"API connection error while prompting GPT model: {e}")
                return None
            except openai.error.InvalidRequestError as e:
                logger.error(f"Invalid request error while prompting GPT model: {e}")
                return None
            except openai.error.AuthenticationError as e:
                logger.error(f"Authentication error while prompting GPT model: {e}")
                return None
            except openai.error.PermissionError as e:
                logger.error(f"Permission error while prompting GPT model: {e}")
                return None
            except openai.error.RateLimitError as e:
                logger.error(f"Rate limit error while prompting GPT model: {e}")
                return None


class Ollama(LLM):
    def __init__(
        self,
        config: dict[str, str | None],
        model: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "json",
    ):
        self.client = OpenAI(api_key=config["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.prompts_path = os.path.join(config["DATA_PATH"], "prompts")
        self.chat_completions = []

    def prompt(
        self, result_id: str, prompts: list[Prompt], add_to_batch: bool = False
    ) -> Optional[str]:
        if add_to_batch:
            raise NotImplementedError("Batch processing is not supported for Ollama")

        messages = []

        for prompt in prompts:
            messages.append({"role": prompt.role, "content": prompt.content})

        logger.info(f"Prompting Ollama model {self.model}")

        chat_completion = self._prompt_model(messages, self.response_format == "json")

        logger.error("Chat completion finished")
        return chat_completion

    @staticmethod
    def run_batch(path: str, dry_run: bool = False) -> Optional[str]:
        raise NotImplementedError("Batch processing is not supported for Ollama")

    def _prompt_model(
        self, messages: list[dict[str, str]], json_response: bool = False
    ) -> str:
        data = {
            "format": "json" if json_response else None,
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": 20000,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(
            "http://localhost:11434/api/chat", headers=headers, json=data
        )

        return response.json()["message"]["content"]
