import argparse
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI()


code_prompt = PromptTemplate.from_template(
    template="Write a very short {language} function that will {task}"
)

test_prompt = PromptTemplate.from_template(
    template="Write a test for the following language {language} code:\n{code}"
)

code_chain = code_prompt | llm | StrOutputParser()

test_chain = (
    code_chain
    | (lambda code: {"code": code, "language": args.language})
    | test_prompt
    | llm
    | StrOutputParser()
)

result = test_chain.invoke(
    {
        "language": args.language,
        "task": args.task,
    }
)


print(result)
