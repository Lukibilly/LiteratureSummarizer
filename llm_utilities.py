import openai
import os
import langchain as lc
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def get_llm(model_name, temperature):
    model_name = model_name
    llm = OpenAI(model_name =   model_name,
                temperature =   temperature)
    return llm

def get_template(template_name):
    with open('templates/' + template_name + '.txt','r') as f:
        template = f.read()
    return template

def make_query_from_faissindex(indexname, query, k):
    faiss_index = FAISS.load_local("literature/" + indexname, OpenAIEmbeddings())
    docs = faiss_index.similarity_search(query, k=k)
    for doc in docs:
        print(doc.metadata["source"], doc.metadata["page"])
    return docs

def run_query_answering_chain(literature, query, llm):
    query_answering_chain = get_query_answering_chain(llm)
    answer = query_answering_chain.run(literature = literature, query=query)
    return answer

def run_page_infoextracter_chain(page, query, llm):
    page_infoextracter_chain = get_page_infoextracter_chain(llm)
    answer = page_infoextracter_chain.run(page = page, query=query)
    return answer

def get_query_answering_chain(llm):
    query_answering_template = get_template('query_answering_template')
    query_answering_prompt = lc.PromptTemplate(input_variables=['literature','query'],template=query_answering_template)
    query_answering_chain = lc.LLMChain(prompt=query_answering_prompt,llm=llm)
    return query_answering_chain

def get_page_infoextracter_chain(llm):
    page_infoextracter_template = get_template('page_infoextracter_template')
    page_infoextracter_prompt = lc.PromptTemplate(input_variables=['page', 'query'],template=page_infoextracter_template)
    page_infoextracter_chain = lc.LLMChain(prompt=page_infoextracter_prompt,llm=llm)
    return page_infoextracter_chain