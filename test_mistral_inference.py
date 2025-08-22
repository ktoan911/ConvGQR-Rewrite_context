import logging
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(".")
sys.path.append("..")


class ConversationalQueryRewriter:
    def __init__(
        self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.3", device: str = None
    ):
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

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Mistral không có pad_token mặc định
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise

    def _padding_seq_to_same_length(
        self, input_ids: List[int], max_pad_length: int, pad_token: int = None
    ):
        if pad_token is None:
            pad_token = self.tokenizer.pad_token_id

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

    def generate_summary_query(
        self, conversation_history: List[str], current_query: str
    ) -> str:
        try:
            inputs = self._prepare_input(conversation_history, current_query)
            with torch.no_grad():
                # Với Causal LM, cần generate từ input
                output_seqs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=False,
                    max_new_tokens=self.max_query_length,  # max_new_tokens thay vì max_length
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Chỉ lấy phần được generate (loại bỏ input)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output_seqs[0][input_length:]
            summary_query = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return summary_query.strip()

        except Exception as e:
            logger.error(f"Lỗi khi generate summary query: {e}")
            return current_query


# Test code
if __name__ == "__main__":
    # Test với conversation history
    conversation_history = [
        "What are the main breeds of goat?",
        "There are many goat breeds including Boer, Angora, Nubian, etc.",
        "Tell me about boer goats",
        "The Boer goat is a breed developed in South Africa for meat production.",
        "What breed of goats is good for meat",
    ]

    current_query = "Are angora goats good for it?"

    try:
        rewriter = ConversationalQueryRewriter()
        result = rewriter.generate_summary_query(conversation_history, current_query)
        print(f"Original query: {current_query}")
        print(f"Rewritten query: {result}")
    except Exception as e:
        print(f"Error: {e}")
