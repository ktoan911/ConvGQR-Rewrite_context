import logging
import sys
from typing import Dict, List

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(".")
sys.path.append("..")

import json
import re
from vllm import LLM, SamplingParams

def to_dict_query(long_string):
    match = re.search(r'\{"query":\s*".*?"\}', long_string)
    if match:
        dict_string = match.group(0)
        try:
            result_dict = json.loads(dict_string)
            return result_dict["query"]
        except Exception:
            return None


def build_rewrite_prompt(history, query):
    history_temp = history[-20:]
    history_str = ""
    for i, turn in enumerate(history_temp):
        if i % 2 == 0:  # user query
            history_str += f"User: {turn}\n"
        else:  # assistant answer
            history_str += f"Assistant: {turn}\n"

    # Prompt dạng one-shot
    template = """
            You are a conversational query rewriting assistant.

            Given:
            - Conversation history (Question/Answer pairs).
            - A new user question.

            Follow the steps **in order**:

            0. Coreference & Entity Linking: Resolve all anaphora/ellipsis before rewriting.
            - Replace pronouns (it, this, they, those, etc.) with the most recent explicit noun phrase or named entity from the conversation context.
            - Replace possessive adjectives (my, your, his, her, their, our, its) by inserting the **explicit object phrase** they refer to (e.g., "their smell" -> "smell of Vietnamese food") — DO NOT invent person names or remove possessives.
            - If it's too difficult to determine, just focus on the user's previous 2 user chat sentences to see what it is.
            1. Question Disambiguation: Rewrite the new question so that it is fully clear and self-contained. Write the new question without any introduction.
            2. Response Expansion: Give a one-sentence response to the new question.
            3. Pseudo Response: You are given a question-and-answer pair, where the answer is not clear. Your goal is to write a long version of the answer based on its given context. The generated answer should be one sentence only and less than 20 words.
            4. Topic Switch: Given a series of question-and-answer pairs, along with a new question, your task is to determine whether the new question continues the discussion on an existing topic or introduces a new topic. Please respond with either "new_topic" or "old_topic" as appropriate.
            5. History Summary: If "old_topic", write a paragraph that summarizes the information in the context. The summary should be short with one sentence for each question answer pair. If "new_topic", skip summary.
            6. Raw Question Repetition: Repeat the original question to avoid forgetting information.
            7. Finally, using all the rewritten/expanded information, convert the new question into a search engine query that can be used to retrieve relevant documents. You MUST keep all proper names in the query. Greetings or polite inquiries SHOULD NOT be edited. The output MUST be placed in a JSON dictionary as follows: {{"query": ""}}

            The output of the current step is the input of the next step. Think step by step, but only show the final JSON at the end.


            ### One-shot Example

            Conversation history:
            User: Who won the NBA finals in 2020?
            Assistant: Lakers.
            User: Who was the MVP?
            Assistant: LeBron James.

            New user question:
            User: Who won in 2021?

            Reasoning:
            0. Coreference & Entity Linking: "Who won in 2021?"
            1. Question Disambiguation: "Who won the NBA finals in 2021?"
            2. Response Expansion: "The MVP was Giannis Antetokounmpo, a star player for the Milwaukee Bucks."
            3. Pseudo Response: "The Milwaukee Bucks won in 2021."
            4. Topic Switch: old_topic.
            5. History Summary: (Question 1/Answer 1) -> Lakers won in 2020. (Question 2/Answer 2) -> MVP was LeBron James IN 2020.
            6. Raw Question Repetition: "Who won in 2021?"
            7. Final query: {{"query": "NBA finals 2021 winner"}}

            ### Your Turn

            Conversation history:
            {history_str}

            New user question:
            {query}

            Reasoning:
        """

    prompt = template.format(history_str=history_str, query=query)

    return prompt


def correct_query(query):
    return f"""
You are a query rewriting assistant.  
Your task is to rewrite the given user query for correct grammar, same language and no spelling mistakes. If there are acronyms and you can understand their meaning in the context of the current chat user, please add them in full form.
 The output MUST be placed in a JSON dictionary as follows: {{"query": ""}}

### Examples

User query: "eather tomorrow"
Rewritten query: {{"query": "weather tomorrow"}}

User query: "capita of France"
Rewritten query: {{"query": "Capital of France"}}

User query: "GDP meaning in domain business development"
Rewritten query: {{"query": "GDP (Gross Domestic Product) meaning in domain business development"}}

### Instruction
Now, rewrite the following query into a semantically complete sentence for retrieval:

User query: "{query}"
Rewritten query:
"""


model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=1.0, 
    top_p=1.0,
    top_k=-1,
    max_tokens=128,
    stop_token_ids=[tokenizer.eos_token_id] 
)
llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            dtype='auto',
        )


class ConversationalQueryRewriter:
    def __init__(self, model_path: str = "ktoan911/ConvGQR", device: str = None):
        self.model_path = model_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # self._load_model()
        self.max_query_length = 32
        self.max_response_length = 64
        self.max_concat_length = 512
        self.use_prefix = True
        # self.model = create_gemini_client("gemini-1.5-flash")

    def _load_model(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise

    def _padding_seq_to_same_length(
        self, input_ids: List[int], max_pad_length: int, pad_token: int = 0
    ):
        padding_length = max_pad_length - len(input_ids)
        attention_mask = []

        if padding_length <= 0:
            attention_mask = [1] * max_pad_length
            input_ids = input_ids[:max_pad_length]
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [pad_token] * padding_length

        return input_ids, attention_mask

    def _prepare_input(
        self, conversation_history: List[str], current_query: str
    ) -> Dict[str, torch.Tensor]:
        flat_concat = []

        if self.use_prefix:
            current_query_text = "question: " + current_query
        else:
            current_query_text = current_query

        # Encode current query
        cur_utt = self.tokenizer.encode(
            current_query_text,
            add_special_tokens=True,
            max_length=self.max_query_length,
        )
        flat_concat.extend(cur_utt)

        first_context = True
        for j in range(len(conversation_history) - 1, -1, -1):
            if j % 2 == 1:  # Response
                max_length = self.max_response_length
            else:  # Query
                max_length = self.max_query_length

            context_text = conversation_history[j]
            if self.use_prefix and first_context:
                context_text = "context: " + context_text
                first_context = False

            utt = self.tokenizer.encode(
                context_text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )

            if len(flat_concat) + len(utt) > self.max_concat_length:
                remaining_length = self.max_concat_length - len(flat_concat) - 1
                if remaining_length > 0:
                    flat_concat += utt[:remaining_length] + [utt[-1]]
                break
            else:
                flat_concat.extend(utt)

        # Padding
        flat_concat, flat_concat_mask = self._padding_seq_to_same_length(
            flat_concat, max_pad_length=self.max_concat_length
        )

        return {
            "input_ids": torch.tensor([flat_concat], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor([flat_concat_mask], dtype=torch.long).to(
                self.device
            ),
        }

    def call_gemini(self, prompts):
        outputs = llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        return results

    def rewrite_one(self, current_querys):
        prompts = [correct_query(current_query) for current_query in current_querys]
        p_llm = self.call_gemini(prompts)
        res = [
            to_dict_query(p) if to_dict_query(p) is not None else current_query
            for p, current_query in zip(p_llm, current_querys)
        ]
        return res

    def rewrite_many(self, conversations, current_querys):
        prompts = [
            build_rewrite_prompt(conversation_history, current_query)
            for conversation_history, current_query in zip(
                conversations, current_querys
            )
        ]
        p_llm = self.call_gemini(prompts)
        res = [
            to_dict_query(p) if to_dict_query(p) is not None else current_query
            for p, current_query in zip(p_llm, current_querys)
        ]
        return res

    def rewrite(
        self, conversation_history: List[str], current_query: str, use_api: bool
    ):
        if len(conversation_history) < 2:
            p_llm = self.call_gemini(correct_query(current_query))

        elif use_api:
            prompt = build_rewrite_prompt(conversation_history, current_query)

            p_llm = self.call_gemini(prompt)

        else:
            return self.generate_summary_query(conversation_history, current_query)

        res = to_dict_query(p_llm)
        if res is None:
            return current_query
        return res

    def generate_summary_query(
        self, conversation_history: List[str], current_query: str
    ) -> str:
        try:
            inputs = self._prepare_input(conversation_history, current_query)
            with torch.no_grad():
                output_seqs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=False,
                    max_length=self.max_query_length,
                    num_return_sequences=1,
                )
            summary_query = self.tokenizer.decode(
                output_seqs[0], skip_special_tokens=True
            )

            return summary_query

        except Exception as e:
            logger.error(f"Lỗi khi generate summary query: {e}")
            return current_query
