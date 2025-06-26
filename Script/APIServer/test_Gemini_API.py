import os
import sys
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'E:\Data\datasets\Video_Datasets\Koala-36M\Code\GeminiAPI\video-describe-3b4a5221da7f.json'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
from langchain_google_vertexai import ChatVertexAI

from langchain.prompts import ChatPromptTemplate


chat = ChatVertexAI(
    model="gemini-2.0-flash-002",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None,
    project_id="video-describe"
    # other params...
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)
messages = prompt.format_messages()
res = chat(messages)
print(res)