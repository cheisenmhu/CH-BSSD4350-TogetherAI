# Programmer: Chris Heise (crheise@icloud.com)
# Course: BSSD 4350 Agile Methodoligies
# Instructor: Jonathan Lee
# Program: Together AI POC
# Purpose: Build a POC using Together AI and Langchain for inclusivity app.
# File: main.py

# Code originally from: 
# URL: https://colab.research.google.com/drive/1RW2yTxh5b9w7F3IrK00Iz51FTO5W01Rx#scrollTo=3H7ZINSIqSyn
# Author: Unknown
# License: Not Listed
# Date Accessed: 19 Sept 2023
# CHANGELOG:
#  - changed prompts/instructions to align with our intended use
#  - 

from together_llm import TogetherLLM
import langchain_prompting as lp
from langchain import PromptTemplate, LLMChain

llm = TogetherLLM(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.1,
    max_tokens=512
)

system_prompt = "You are an advanced assistant that excels at rewriting text to be more inclusive. If text contains any sexist, racist, ableist, homophobic, or otherwise non-inclusive language, you re-write it to be inclusive to all people. If you think the text is already inclusive, then just say there's nothing to change."
instruction = "Rewrite the following text to be more inclusive:\n\n {text}"
template = lp.get_prompt(instruction, system_prompt)

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
text = "I think Mary is transgendered."
output = llm_chain.run(text)

print(output)
# Response:
"""I think Mary identifies as transgender.

It's important to use language that is respectful and inclusive of all people, regardless of their gender identity or expression. Using outdated or derogatory terms can be hurtful and contribute to a negative and discriminatory environment. By using the term "identifies as transgender," we are using language that is more inclusive and respectful of Mary's identity."""

# TODO:
#   In order to use embeddings/context,
#    1. Create our own together embeddings class from langchain.embeddings.base -> base class
#    2. Tinker with our prompt/instructions to include {context} 
#    3. Figure out how to figure out how to use a RetrievalQA with our custom embeddings and llm object
