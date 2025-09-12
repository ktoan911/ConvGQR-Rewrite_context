import re
import time

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
model_name = "BeastyZ/Qwen2.5-3B-ConvSearch-R1-TopiOCQA"


def get_rewrite_text(text):
    matches = re.findall(r"<rewrite>(.*?)</rewrite>", text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    return text


class Rewrite:
    def __init__(self):
        self.sampling_params = SamplingParams(
                            temperature=0.7,
                            max_tokens=4096
                        )
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            enforce_eager=False,
            gpu_memory_utilization=0.85,
            dtype='float16',
        )
    def rewrite_query(self, histories, query):
        start = time.time()
        print(f"start {start}")

        history_chat = ""
        for i, h in enumerate(histories[-10:]):
            history_chat += f"Q{i+1}: {h}\n" if i % 2 == 0 else f"A{i+1}: {h}\n"
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
        return get_rewrite_text(
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        )

    def rewrite_queries(self, histories, queries):
        history_chat_strs = []
        for history in histories:
            history_chat = ""
            for i, h in enumerate(history[-10:]):
                history_chat += f"Q{i+1}: {h}\n" if i % 2 == 0 else f"A{i+1}: {h}\n"
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
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = [output.outputs[0].text for output in outputs]
        return [get_rewrite_text(res) for res in results]
