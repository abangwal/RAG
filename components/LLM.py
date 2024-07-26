"""
Contain Classes for LLM inference for RAG pipeline.
"""

### ** Make input output tokens as class properties **

from openai import OpenAI
from .utils import count_tokens
import json


class rLLM:
    def __init__(self, llm_name: str, api_key: str) -> None:
        self.llm_name = llm_name
        self.llm_client = OpenAI(
            api_key=api_key, base_url="https://api.together.xyz/v1"
        )
        with open("components/prompts.json", "r") as file:
            SysPrompt = json.load(file)["SysPrompt"]
        self.sys_prompt = SysPrompt

    def generate_rag_response(self, context: str, prompt: str, message_history):
        """
        Generates a natural language response for user query(prompt) based on provided
        context and message history, in Q&A style.
        """

        system_prompt = self.sys_prompt

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for message in message_history[-6:-1]:
            if message["role"] == "assistant":
                messages.append({"role": "assistant", "content": message["content"]})
            else:
                messages.append(message)

        messages.append(
            {"role": "user", "content": f"CONTEXT:\n{context}QUERY:\n{prompt}"},
        )

        stream = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=True,
        )

        output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                output += content
                yield 0, content

        input_token_count = count_tokens(
            string="\n".join([i["content"] for i in messages])
        )

        output_token_count = count_tokens(string=output)

        yield 1, (output, input_token_count, output_token_count)

    def HyDE(self, query: str, message_history):
        """
        Rephare/rewrite the user query to include more semantics, hence improving the
        semantic search based retreival.
        """

        system_prompt = """You are an AI assistant specifically designed to generate hypothetical answers for semantic search. Your primary function is to create concise Maximum 100-150 words, informative, and relevant responses to user queries. Make sure to capture the original intent of the user query (by including keywords present in user query) as these responses will be used to generate embeddings for improved semantic search results.
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for message in message_history[-6:-1]:
            if message["role"] == "assistant":
                messages.append({"role": "assistant", "content": message["content"]})
            else:
                messages.append(message)

        messages.append(
            {"role": "user", "content": f"\n\nQUERY:\n{query}"},
        )

        response = self.llm_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=messages,
            max_tokens=500,
        )
        response = response.choices[0].message.content
        return response

    ### NOT IN USE

    def generate_rag_chat_response(self, context: str, prompt: str, message_history):
        """
        NOT IN USE CURRENTLY
        Generates a natural language response for user query(prompt) based on provided
        context and message history, in Q&A style.
        """

        system_prompt = """You are a helpful legal compliance CHAT assistant designed to answer and resolve user query in chat format hence quick and *small* responses.

Instructions:
1. Use the provided CONTEXT to inform your responses, citing specific parts when relevant.
2. If unable to answer the QUERY, politely inform the user and suggest what additional information might help.
3. Give a small/chatty format response.
4. Try to give decisive responses that can help user to make informed decision. 
5. Format responses for readability, include bold words to give weightage.

Don't add phrases like "According to provided context.." etcectra. """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for message in message_history[-6:-1]:
            if message["role"] == "assistant":
                messages.append({"role": "assistant", "content": message["content"]})
            else:
                messages.append(message)

        messages.append(
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY:\n{prompt}"},
        )

        stream = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=True,
        )

        output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                output += content
                yield 0, content

        input_token_count = count_tokens(
            string="\n".join([i["content"] for i in messages])
        )

        output_token_count = count_tokens(string=output)

        yield 1, (output, input_token_count, output_token_count)

    def rephrase_query(self, query: str, message_history):
        """
        NOT IN USE CURRENTLY
        Rephare/rewrite the user query to include more semantics, hence improving the
        semantic search based retreival.
        """

        system_prompt = """You are an AI assistant specifically designed to rewrite the user QUERY for semantic search. Your primary function is to create more comprehensive, semantically rich query *while maintaining the original intent of the user*. These responses will be used to generate embeddings for improved semantic search results.
Do not include any other comments or text, other than the repharased/rewritten query.
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for message in message_history[-6:-1]:
            if message["role"] == "assistant":
                messages.append({"role": "assistant", "content": message["content"]})
            else:
                messages.append(message)

        messages.append(
            {"role": "user", "content": f"\n\nQUERY:\n{query}"},
        )

        response = self.llm_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=messages,
        )
        response = response.choices[0].message.content
        return response
