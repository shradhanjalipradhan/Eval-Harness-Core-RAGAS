import os
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.transformers import TransformersChatGenerator

def prepare_document_store():
    print("Initializing Document Store...")
    document_store = InMemoryDocumentStore()
    
    print("Fetching Seven Wonders dataset from Hugging Face...")
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    
    print("Generating embeddings and indexing documents...")
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])
    
    return document_store

def build_rag_pipeline(document_store):
    print("Building Haystack RAG Pipeline...")
    
    # 1. Initialize Components
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store)
    
    template = [
        ChatMessage.from_user(
            """Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:"""
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
    
    # Using local/free inference via Transformers (Qwen-3)
    chat_generator = TransformersChatGenerator(model="Qwen/Qwen3-4B-Instruct-2507")
    
    # 2. Build Pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", chat_generator)
    
    # 3. Connect Components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    
    return pipeline

def main():
    document_store = prepare_document_store()
    pipeline = build_rag_pipeline(document_store)
    
    question = "What does Rhodes Statue look like?"
    print(f"\nRunning Query: '{question}'...\n")
    
    response = pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question}
    })
    
    print("=== AI Response ===")
    print(response["llm"]["replies"][0].text)
    print("===================")

if __name__ == "__main__":
    main()
