#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install langchain-community==0.2.4 langchain==0.2.3 faiss-cpu==1.8.0 unstructured==0.14.5 unstructured[pdf]==0.14.5 transformers==4.41.2 sentence-transformers==3.0.1')


# **Importing the Dependencies**

# In[6]:


import os

from langchain_community.llms import Ollama
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# In[7]:


# loading the LLM
llm = Ollama(
    model="llama3:instruct",
    temperature=0
)


# In[8]:


# loading the document
loader = UnstructuredFileLoader("NIPS-2017-attention-is-all-you-need-Paper.pdf")
documents = loader.load()


# In[9]:


# create document chunks
text_splitter = CharacterTextSplitter(separator="/n",
                                      chunk_size=1000,
                                      chunk_overlap=200)


# In[11]:


text_chunks = text_splitter.split_documents(documents)


# In[12]:


# loading the vector embedding model
embeddings = HuggingFaceEmbeddings()


# In[15]:


knowledge_base = FAISS.from_documents(text_chunks, embeddings)


# In[16]:


# retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())


# In[28]:


from langchain_community.llms import Ollama

llm = Ollama(model="llama3:latest")
llm.invoke("Tell me a joke"


# In[26]:


question = "What is this document about?"
response = qa_chain.invoke({"query": question})
print(response["result"])


# In[27]:


question = "What is the architecture discussed in the model?"
response = qa_chain.invoke({"query": question})
print(response["result"])


# In[ ]:





# In[ ]:




