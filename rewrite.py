import logging
import sys
from typing import Dict, List

import google.generativeai as genai
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(".")
sys.path.append("..")


def build_rewrite_prompt(history, query):
    # Chuyển history thành chuỗi có format rõ ràng
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
            - Conversation history (Q/A pairs).
            - A new user question.

            Follow the steps **in order**:

            1. Topic Switch: Given a series of question-and-answer pairs, along with a new question, your task is to determine whether the new question continues the discussion on an existing topic or introduces a new topic. Please respond with either "new_topic" or "old_topic" as appropriate.
            2. If "old_topic", write a paragraph that summarizes the information in the context. The summary should be short with one sentence for each question answer pair. If "new_topic", skip summary.
            3. Question Disambiguation: Rewrite the new question so that it is fully clear and self-contained. Write the new question without any introduction.
            4. Response Expansion: Give a one-sentence response to the new question.
            5. Pseudo Response: After the above steps you have a series of question-and-answer pairs as context along with a new question, your task is to generate a set of search queries based on the relevancy between the new question and the relevant passage and also rely on the given context. The output format should be in a list with indexes e.g., 1. 2. 3.
            6. Finally, using all the rewritten/expanded information, convert the new question into a search engine query that can be used to retrieve relevant documents. The output should be placed in a JSON dictionary as follows: "query": ""

            Think step by step, but only show the final JSON at the end.

            ---

            ### One-shot Example

            Conversation history:
            User: Who won the NBA finals in 2020?
            Assistant: Lakers.
            User: Who was the MVP?
            Assistant: LeBron James.

            New user question:
            User: Who won in 2021?

            Reasoning:
            1. TS: old_topic.
            2. HS: (Q1/A1) -> Lakers won in 2020. (Q2/A2) -> MVP was LeBron James.
            3. QD: "Who won the NBA finals in 2021?"
            4. RE: "The MVP was LeBron James, a star player for the Lakers."
            5. PR: "The Milwaukee Bucks won in 2021."
            6. Final query: "query": "NBA finals 2021 winner"

            ---

            ### Your Turn

            Conversation history:
            {{history_str}}

            New user question:
            {{query}}

            Reasoning:
        """

    # Bây giờ, prompt sẽ sử dụng đúng giá trị history_str đã tính ở trên
    # và giá trị query được truyền vào hàm.
    prompt = template.format(history_str=history_str, query=query)

    return prompt


class ConversationalQueryRewriter:
    def __init__(self, model_path: str = "ktoan911/ConvGQR", device: str = None):
        self.model_path = model_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()
        self.max_query_length = 32
        self.max_response_length = 64
        self.max_concat_length = 512
        self.use_prefix = True
        genai.configure(api_key="AIzaSyDTKjpeTjoPUKDrkkg0Xk1BbSfb60WOAmg")
        self.model = genai.GenerativeModel("gemini-1.5-flash")

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

    def call_gemini(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

    def rewrite(
        self, conversation_history: List[str], current_query: str, use_api: bool
    ):
        if use_api:
            prompt = build_rewrite_prompt(conversation_history, current_query)
            return self.call_gemini(prompt)
        else:
            return self.generate_summary_query(conversation_history, current_query)

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


# t = [
#     "What are the main breeds of goat?",
#     "Abaza...Zhongwei",
#     "Tell me about boer goats",
#     "The Boer goat is a breed of goat that was developed in South Africa in the early 1900s for meat production. Their name is derived from the Afrikaans (Dutch) word boer, meaning farmer.",
#     "What breed of goats is good for meat",
#     "Before Boer goats became available in the United States in the late 1980s, Spanish goats were the standard meat goat breed, especially in the South. These goats are descendants of the goats brought by Spanish explorers, making their way to the United States via Mexico.",
# ]
# rewriter = ConversationalQueryRewriter()

# t1 = "Are angora goats good for it?"
# res = rewriter.generate_summary_query(t, t1)
# print(res)
# model = SentenceTransformer("all-MiniLM-L6-v2")


# def semantic_similarity(sentence1: str, sentence2: str) -> float:

#     emb1 = model.encode(sentence1, convert_to_tensor=True)
#     emb2 = model.encode(sentence2, convert_to_tensor=True)

#     similarity = util.cos_sim(emb1, emb2)

#     return similarity.item()
