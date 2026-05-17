from langchain_community.document_loaders import WebBaseLoader

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

model = ChatOllama(model="llama3.2:1b")

load_dotenv()

prompt = PromptTemplate(
    template= "write the answer the following questions -\n {question} from the following text -\n{text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"

loader = WebBaseLoader(url) # we can pass multiple url as list 

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)


# below are direct use of  model request 

chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))



