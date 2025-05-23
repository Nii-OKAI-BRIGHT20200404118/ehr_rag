import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from httpx import Timeout, RequestError, HTTPStatusError, TimeoutException
import asyncio
import tenacity
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

@dataclass
class OllamaConfig:
    llm_model: str = "qwen2:1.5b"
    #llm_model: str = "tinyllama"
    #llm_model: str = "phi"
    #llm_model: str = "gemma2:2b"
    #llm_model: str = "dolphin-phi"
    embedding_model: str = "nomic-embed-text:v1.5"
    api_base_url: str = "http://localhost:11434"
    chunk_size: int = 256  # Reduced chunk size
    chunk_overlap: int = 5  # Reduced overlap
    timeout: float = 60.0  # Increased timeout
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrent_requests: int = 3  # Limit concurrent requests

class EHRDocument:
    def __init__(self, patient_id: str, records: List[Dict]):
        self.patient_id = patient_id
        self.records = records

    def to_string(self) -> str:
        return json.dumps({"patient_id": self.patient_id, "records": self.records}, indent=2)

class EHRSummarizer:
    def __init__(self, config: OllamaConfig):
        try:
            timeout = Timeout(config.timeout)
            
            self.llm = Ollama(model=config.llm_model, base_url=config.api_base_url, timeout=timeout)
            self.embed_model = OllamaEmbedding(model_name=config.embedding_model, base_url=config.api_base_url, timeout=timeout)
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = config.chunk_size
            Settings.chunk_overlap = config.chunk_overlap
            
            self.node_parser = SimpleNodeParser.from_defaults()
            self.config = config
            self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        except Exception as e:
            logging.error(f"Failed to initialize EHRSummarizer: {e}")
            raise RuntimeError(f"Initialization error: {e}")

    def create_index(self, ehr_documents: List[EHRDocument]):
        try:
            documents = [Document(text=doc.to_string(), id_=doc.patient_id) for doc in ehr_documents]
            nodes = self.node_parser.get_nodes_from_documents(documents)
            if not nodes:
                raise ValueError("No nodes were created from the documents")
            self.index = VectorStoreIndex(nodes)
            self.query_engine = self.index.as_query_engine()
            logging.info("Index created successfully")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            raise RuntimeError(f"Error creating index: {e}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type((TimeoutException, HTTPStatusError, RequestError)),
        reraise=True
    )
    async def _query_with_retry(self, prompt: str) -> str:
        async with self.semaphore:
            start_time = time.time()
            response = await self.query_engine.aquery(prompt)
            end_time = time.time()
            logging.info(f"Query execution time: {end_time - start_time:.2f} seconds")
            return response.response if hasattr(response, 'response') else str(response)

    async def summarize_async(self, task: str, patient_id: Optional[str] = None) -> str:
        try:
            prompt = f"For patient {patient_id}, {task}" if patient_id else task
            return await self._query_with_retry(prompt)
        except TimeoutException as timeout_err:
            logging.error(f"Request timed out after retries: {timeout_err}")
            return "The request timed out. Please try again later."
        except HTTPStatusError as http_err:
            logging.error(f"HTTP error occurred after retries: {http_err}")
            return f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}"
        except RequestError as req_err:
            logging.error(f"Request error occurred after retries: {req_err}")
            return f"Request error occurred: {req_err}"
        except Exception as e:
            logging.error(f"An unexpected error occurred during summarization: {e}")
            return f"An unexpected error occurred: {e}"

def load_sample_ehr_data() -> List[EHRDocument]:
    return [
        EHRDocument("P001", [
            {"date": "2023-01-15", "type": "visit", "notes": "Patient complains of persistent cough for 2 weeks. Prescribed antibiotics."},
            {"date": "2023-02-01", "type": "lab", "result": "Blood test shows elevated white blood cell count."},
            {"date": "2023-02-10", "type": "visit", "notes": "Follow-up visit. Cough has improved. Continue current treatment."}
        ]),
        EHRDocument("P002", [
            {"date": "2023-03-01", "type": "visit", "notes": "Annual check-up. Patient reports feeling well. No significant issues."},
            {"date": "2023-03-15", "type": "lab", "result": "Cholesterol levels slightly elevated. Recommend dietary changes."}
        ])
    ]

async def process_query(summarizer: EHRSummarizer, task: str, patient_id: Optional[str]) -> str:
    try:
        return await summarizer.summarize_async(task, patient_id)
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Error: {str(e)}"

async def main():
    config = OllamaConfig(
        #llm_model="tinyllama",
        llm_model="qwen2:1.5b",
        #llm_model="phi",
        #llm_model: str = "dolphin-phi",
        #llm_model: str = "gemma2:2b",
        embedding_model="nomic-embed-text:v1.5",
        api_base_url="http://localhost:11434",
        chunk_size=256,
        chunk_overlap=5,
        timeout=60.0,
        max_retries=3,
        retry_delay=2.0,
        max_concurrent_requests=3
    )

    summarizer = EHRSummarizer(config)
    ehr_data = load_sample_ehr_data()

    summarizer.create_index(ehr_data)

    logging.info("EHR Summarization System initialized. You can start asking for summaries.")
    print("Type 'quit' to exit.")

    while True:
        print("\nChoose a summarization task:")
        print("1. Summarize patient history")
        print("2. List all medications")
        print("3. Summarize lab results")
        print("4. Custom query")
        print("5. Batch process all patients")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice.lower() == 'quit':
            break

        if choice == "5":
            tasks = [
                "Provide a concise summary of the patient's medical history.",
                "List all medications prescribed to the patient.",
                "Summarize all lab results for the patient."
            ]
            patient_ids = [doc.patient_id for doc in ehr_data]
            
            start_time = time.time()
            results = await asyncio.gather(*[process_query(summarizer, task, patient_id) 
                                             for task in tasks 
                                             for patient_id in patient_ids])
            end_time = time.time()
            
            print("\nBatch Processing Results:")
            for i, result in enumerate(results):
                print(f"\nTask {(i // len(patient_ids)) + 1}, Patient {patient_ids[i % len(patient_ids)]}:")
                print(result)
            
            print(f"\nTotal batch processing time: {end_time - start_time:.2f} seconds")
        else:
            patient_id = input("Enter patient ID (or press Enter for all patients): ")

            if choice == "1":
                task = "Provide a concise summary of the patient's medical history."
            elif choice == "2":
                task = "List all medications prescribed to the patient."
            elif choice == "3":
                task = "Summarize all lab results for the patient."
            elif choice == "4":
                task = input("Enter your custom query: ")
            else:
                print("Invalid choice. Please try again.")
                continue

            start_time = time.time()
            summary = await process_query(summarizer, task, patient_id if patient_id else None)
            end_time = time.time()
            
            print(f"\nSummary: {summary}")
            print(f"Query execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())