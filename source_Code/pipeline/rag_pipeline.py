from source_Code.retrieval.retriever import (
    DualRetriever
)

from source_Code.llm.prompt_builder import (
    PromptBuilder
)

from source_Code.llm.generator import (
    Generator
)


retriever = DualRetriever()


while True:

    question = input(

        "\nAsk Medical Question : "

    )

    if question.lower() == "exit":

        break

    retrieval = retriever.retrieve(

        question

    )

    prompt = PromptBuilder.build_prompt(

        question,

        retrieval["expert"]

    )

    answer = Generator.generate(

        prompt

    )

    print("\n")

    print("=" * 60)

    print("ANSWER")

    print("=" * 60)

    print(answer)

    print("=" * 60)