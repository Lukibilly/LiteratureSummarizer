import openai
import os
import langchain as lc
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from llm_utilities import *

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key

query = """ What different ways exist to balance the reconstruction and regularisation KL divergence term of the ELBO loss in Variational Autoencoders?""" 

llm_page_extractor = get_llm('text-curie-001', 1)
llm_query_answering = get_llm('text-davinci-003', 0.8)

docs = make_query_from_faissindex('BALiterature', query, 20)

literature = ''

for doc in docs:
    newinfo =  str("source:" + doc.metadata["source"]) + "," + "page:" + str(doc.metadata["page"]) + "content:" + run_page_infoextracter_chain(doc.page_content, query, llm_page_extractor)
    literature += newinfo
    print(newinfo + "\n")
    #literature += str("source:" + doc.metadata["source"]) + "," + "page:" + str(doc.metadata["page"]) + "content:" + doc.page_content

print(run_query_answering_chain(literature, query, llm_query_answering))