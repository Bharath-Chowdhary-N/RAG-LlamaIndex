#objective: CreateRAG with LlamaIndex

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
load_dotenv()


class RAG():
        def __init__(self,query):
              self.pdf_folder="data"
              self.create_metadata_of_pdf()
              self.create_VectorStoreIndex()
              self.retriever()
              self.get_response(query)
        def create_metadata_of_pdf(self):
              self.documents=SimpleDirectoryReader(self.pdf_folder).load_data() 
        def create_VectorStoreIndex(self):
              self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True) 
              print(self.index)
        def retriever(self):
              self.retriever=VectorIndexRetriever(index=self.index, similarity_top_k=4)
              self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.40)
              self.query_engine = RetrieverQueryEngine(retriever=self.retriever, node_postprocessors=[self.postprocessor])
        def get_response(self,query):
              self.response = self.query_engine.query(query)
              print(self.response)

if __name__=="__main__":
   RAG_instance=RAG("What is denselens")