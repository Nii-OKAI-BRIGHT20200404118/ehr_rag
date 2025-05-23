import json
import logging
from typing import List, Dict
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, Settings, Document

#from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

logging.basicConfig(level=logging.INFO)

@dataclass
class OllamaConfig:
    llm_model: str = "gemma2"
    embedding_model: str = "nomic-embed-text:v1.5"
    api_base_url: str = "http://localhost:11434"
    chunk_size: int = 1024
    chunk_overlap: int = 20

class EHRDocument:
    def __init__(self, patient_id: str, records: List[Dict]):
        self.patient_id = patient_id
        self.records = records

    def to_string(self) -> str:
        return json.dumps({"patient_id": self.patient_id, "records": self.records}, indent=2)

class EHRSummarizer:
    def __init__(self, config: OllamaConfig):
        self.config = config
        # Create Ollama LLM and Embedding models
        self.llm = Ollama(model=config.llm_model, base_url=config.api_base_url)
        self.embed_model = OllamaEmbedding(model_name=config.embedding_model, base_url=config.api_base_url)
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = config.chunk_size
        Settings.chunk_overlap = config.chunk_overlap
        
        # Create a node parser for chunking
        self.node_parser = SimpleNodeParser.from_defaults()

    def create_index(self, ehr_documents: List[EHRDocument]):
        # Convert EHR documents to LlamaIndex Document objects
        documents = [
            Document(text=doc.to_string(), id_=doc.patient_id)
            for doc in ehr_documents
        ]
        
        # Create nodes from documents
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        # Create an index from the nodes
        self.index = VectorStoreIndex(nodes)
        
        # Create a query engine
        self.query_engine = self.index.as_query_engine()

    def summarize(self, task: str, patient_id: str = None) -> str:
        if patient_id:
            prompt = f"For patient {patient_id}, {task}"
        else:
            prompt = task

        response = self.query_engine.query(prompt)
        return response.response

def load_sample_ehr_data() -> List[EHRDocument]:
    # This is a mock function. In a real scenario, you'd load actual EHR data.
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

def main():
    # Define configuration
    config = OllamaConfig(
        llm_model="gemma2",
        embedding_model="nomic-embed-text:v1.5",
        api_base_url="http://localhost:11434",
        chunk_size=1024,
        chunk_overlap=20
    )

    # Initialize the EHR summarizer with the configuration
    summarizer = EHRSummarizer(config)

    # Load sample EHR data
    ehr_data = load_sample_ehr_data()

    # Create index from EHR data
    summarizer.create_index(ehr_data)

    logging.info("EHR Summarization System initialized. You can start asking for summaries.")
    print("Type 'quit' to exit.")

    while True:
        print("\nChoose a summarization task:")
        print("1. Summarize patient history")
        print("2. List all medications")
        print("3. Summarize lab results")
        print("4. Custom query")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice.lower() == 'quit':
            break

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

        try:
            summary = summarizer.summarize(task, patient_id if patient_id else None)
            print(f"\nSummary: {summary}")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()