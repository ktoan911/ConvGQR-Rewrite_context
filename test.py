def build_rewrite_prompt(history, query):
    # Chuyển history thành chuỗi có format rõ ràng
    history_temp = history[-20:]
    history_str = ""
    for i, turn in enumerate(history_temp):
        if i % 2 == 0:  # user query
            history_str += f"User: {turn}\n"
        else:  # assistant answer
            history_str += f"Assistant: {turn}\n"
    print(query)

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
            {history_str}

            New user question:
            {query}

            Reasoning:
        """

    prompt = template.format(history_str=history_str, query=query)

    return prompt


print(
    build_rewrite_prompt(
        [
            "Who won the NBA finals in 2020?",
            "Lakers.",
            "Who was the MVP?",
            "LeBron James.",
        ],
        "Who won in 2021?",
    )
)
