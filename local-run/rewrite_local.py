import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "BeastyZ/Llama3.2-3B-ConvSearch-R1-TopiOCQA"

# Load tokenizer và model 1 lần


class Rewrite:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  #
            device_map="cuda",
        )

    def rewrite_query(self, histories, query):
        start = time.time()
        print(f"start {start}")

        # Ghép lịch sử hội thoại ngắn gọn
        history_chat = ""
        for i, h in enumerate(histories[-10:]):
            history_chat += f"Q{i}: {h}\n" if i % 2 == 0 else f"A{i}: {h}\n"
        example = f"""Given a query and its context, you must first
                    think about the reasoning process in the mind to
                    decontextualize the query by resolving coreference
                    and omission issues. Then, provide the user
                    with a rewrite that retains its original meaning
                    and is as informative as possible to help search
                    engines retrieve relevant documents effectively. The
                    reasoning process and rewrite should be enclosed
                    within <think> </think> and <rewrite> </rewrite>
                    tags, respectively, i.e., <think> reasoning process
                    here </think>\n<rewrite> rewrite here </rewrite>.
                    ### Context Begin ###
                    {history_chat}
                    ### Context End ###
                    Query: {query}
                    Rewrite:"""

        inputs = self.tokenizer(example, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
        )

        print(f"end {time.time() - start:.2f} s")
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def rewrite_queries(self, histories, queries):
        history_chat_strs = []
        for history in histories:
            history_chat = ""
            for i, h in enumerate(history[-10:]):
                history_chat += f"Q{i}: {h}\n" if i % 2 == 0 else f"A{i}: {h}\n"
            history_chat_strs.append(history_chat)
        prompts = [
            f"""Given a query and its context, you must first
                    think about the reasoning process in the mind to
                    decontextualize the query by resolving coreference
                    and omission issues. Then, provide the user
                    with a rewrite that retains its original meaning
                    and is as informative as possible to help search
                    engines retrieve relevant documents effectively. The
                    reasoning process and rewrite should be enclosed
                    within <think> </think> and <rewrite> </rewrite>
                    tags, respectively, i.e., <think> reasoning process
                    here </think>\n<rewrite> rewrite here </rewrite>.
                    ### Context Begin ###
                    {history_chat}
                    ### Context End ###
                    Query: {query}
                    Rewrite:"""
            for history_chat, query in zip(history_chat_strs, queries)
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate cho cả batch
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
        )
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return results
