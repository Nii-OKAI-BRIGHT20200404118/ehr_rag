import gradio as gr
import logging
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import csv
import json
from datetime import datetime
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from httpx import Timeout
import chromadb
import asyncio
import tenacity
import time
from concurrent.futures import ThreadPoolExecutor

#from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)

@dataclass
class OllamaConfig:
    llm_model: str = "qwen2:1.5b"
    #llm_model: str = "gemma2:2b"
    #llm_model: str = "phi"
    #llm_model: str = "tinyllama"
    embedding_model: str = "nomic-embed-text:v1.5"
    api_base_url: str = "http://localhost:11434"
    # chunk_size: int = 512
    # chunk_overlap: int = 10
    # timeout: float = 30.0
    chunk_size: int = 256  # Reduced chunk size
    chunk_overlap: int = 5  # Reduced overlap
    timeout: float = 60.0  # Increased timeout
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrent_requests: int = 3  # Limit concurrent requests

class EHRDocument:
    def __init__(self, patient_id: str, birthdate: str, deathdate: str, gender: str, race: str, ethnicity: str, healthcare_expenses: float, healthcare_coverage: float):
        self.patient_id = patient_id
        self.birthdate = birthdate
        self.deathdate = deathdate
        self.gender = gender
        self.race = race
        self.ethnicity = ethnicity
        self.healthcare_expenses = healthcare_expenses
        self.healthcare_coverage = healthcare_coverage
        self.conditions = []
        self.encounters = []
        self.medications = []
        self.procedures = []

class SyntheaParser:
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)

    def parse_data(self) -> List[EHRDocument]:
        ehr_documents = {}
        
        # Parse patients
        patients_file = self.data_dir / 'patients.csv'
        with open(patients_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['Id']
                ehr_documents[patient_id] = EHRDocument(
                    patient_id=patient_id,
                    birthdate=row['BIRTHDATE'],
                    deathdate=row['DEATHDATE'],
                    gender=row['GENDER'],
                    race=row['RACE'],
                    ethnicity=row['ETHNICITY'],
                    healthcare_expenses=float(row['HEALTHCARE_EXPENSES']),
                    healthcare_coverage=float(row['HEALTHCARE_COVERAGE'])
                )

        self._parse_conditions(ehr_documents)
        self._parse_encounters(ehr_documents)
        self._parse_medications(ehr_documents)
        self._parse_procedures(ehr_documents)

        return list(ehr_documents.values())

    def _parse_conditions(self, ehr_documents):
        conditions_file = self.data_dir / 'conditions.csv'
        with open(conditions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['PATIENT']
                if patient_id in ehr_documents:
                    ehr_documents[patient_id].conditions.append({
                        'start': row['START'],
                        'stop': row['STOP'],
                        'code': row['CODE'],
                        'description': row['DESCRIPTION']
                    })

    def _parse_encounters(self, ehr_documents):
        encounters_file = self.data_dir / 'encounters.csv'
        with open(encounters_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['PATIENT']
                if patient_id in ehr_documents:
                    ehr_documents[patient_id].encounters.append({
                        'id': row['Id'],
                        'start': row['START'],
                        'stop': row['STOP'],
                        'type': row['ENCOUNTERCLASS'],
                        'description': row['DESCRIPTION'],
                        'reason_code': row['REASONCODE'],
                        'reason_description': row['REASONDESCRIPTION']
                    })

    def _parse_medications(self, ehr_documents):
        medications_file = self.data_dir / 'medications.csv'
        with open(medications_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['PATIENT']
                if patient_id in ehr_documents:
                    ehr_documents[patient_id].medications.append({
                        'start': row['START'],
                        'stop': row['STOP'],
                        'code': row['CODE'],
                        'description': row['DESCRIPTION'],
                        'dispenses': int(row['DISPENSES']),
                        'total_cost': float(row['TOTALCOST'])
                    })

    def _parse_procedures(self, ehr_documents):
        procedures_file = self.data_dir / 'procedures.csv'
        with open(procedures_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['PATIENT']
                if patient_id in ehr_documents:
                    ehr_documents[patient_id].procedures.append({
                        'date': row['DATE'],
                        'code': row['CODE'],
                        'description': row['DESCRIPTION'],
                        'base_cost': float(row['BASE_COST']),
                        'reason_code': row['REASONCODE'],
                        'reason_description': row['REASONDESCRIPTION']
                    })

class EHRSummarizer:
    def __init__(self, config: OllamaConfig, persist_dir: str = "./chroma_db"):
        try:
            timeout = Timeout(config.timeout)
            
            self.llm = Ollama(model=config.llm_model, base_url=config.api_base_url, timeout=timeout)
            self.embed_model = OllamaEmbedding(model_name=config.embedding_model, base_url=config.api_base_url, timeout=timeout)
            
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = config.chunk_size
            Settings.chunk_overlap = config.chunk_overlap
            
            self.node_parser = SimpleNodeParser.from_defaults()
            self.persist_dir = Path(persist_dir)
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

            
            self.chroma_collection = self.chroma_client.get_or_create_collection("ehr_records")
            
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Check if the index already exists
            if self.chroma_collection.count() > 0:
                logging.info("Loading existing index...")
                self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            else:
                logging.info("No existing index found. You need to create one using create_index().")
                self.index = None
            
            self.query_engine = self.index.as_query_engine() if self.index else None
            
        except Exception as e:
            logging.error(f"Failed to initialize EHRSummarizer: {e}")
            raise RuntimeError(f"Initialization error: {e}")

    def create_index(self, ehr_documents: List[EHRDocument]):
        if self.index is not None:
            logging.info("Index already exists. Skipping index creation.")
            return

        try:
            logging.info("Creating new index...")
            documents = [
                Document(text=self.document_to_string(doc), id_=doc.patient_id)
                for doc in ehr_documents
            ]
            nodes = self.node_parser.get_nodes_from_documents(documents)
            if not nodes:
                raise ValueError("No nodes were created from the documents")
            
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            self.query_engine = self.index.as_query_engine()
            logging.info("Index created successfully")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            raise RuntimeError(f"Error creating index: {e}")

    def document_to_string(self, doc: EHRDocument) -> str:
        patient_info = f"Patient ID: {doc.patient_id}\n"
        patient_info += f"Birthdate: {doc.birthdate}\n"
        patient_info += f"Gender: {doc.gender}\n"
        patient_info += f"Race: {doc.race}\n"
        patient_info += f"Ethnicity: {doc.ethnicity}\n"
        patient_info += f"Healthcare Expenses: ${doc.healthcare_expenses:.2f}\n"
        patient_info += f"Healthcare Coverage: ${doc.healthcare_coverage:.2f}\n\n"

        conditions = "\n".join([f"{c['start']} - {c['description']} (Code: {c['code']})" for c in doc.conditions])
        encounters = "\n".join([f"{e['start']} - {e['description']} (Type: {e['type']}, Reason: {e['reason_description']})" for e in doc.encounters])
        medications = "\n".join([f"{m['start']} to {m['stop']} - {m['description']} (Dispenses: {m['dispenses']}, Total Cost: ${m['total_cost']:.2f})" for m in doc.medications])
        procedures = "\n".join([f"{p['date']} - {p['description']} (Cost: ${p['base_cost']:.2f}, Reason: {p['reason_description']})" for p in doc.procedures])

        return f"{patient_info}Conditions:\n{conditions}\n\nEncounters:\n{encounters}\n\nMedications:\n{medications}\n\nProcedures:\n{procedures}"

    def summarize(self, task: str, patient_id: str) -> str:
        try:
            prompt = (
                f"Based on the electronic health records for patient {patient_id}, please {task}\n"
                f"Only provide information for the specified patient ID: {patient_id}. "
                f"Consider the patient's age, gender, race, ethnicity, and healthcare expenses/coverage when relevant. "
                f"Provide a concise, well-structured response using complete sentences. "
                f"Include specific dates, codes, costs, and descriptions where appropriate. "
                f"If there's not enough information to answer fully, mention what's missing."
            )
            
            response = self.query_engine.query(prompt)
            
            # Additional check to ensure we're only returning information for the correct patient
            if patient_id not in response.response:
                return f"No information found for patient {patient_id}. Please check the patient ID and try again."
            
            return response.response if hasattr(response, 'response') else str(response)
        except Exception as e:
            logging.error(f"An error occurred during summarization: {e}")
            return f"I'm sorry, but an error occurred: {e}. Please try again or contact support for assistance."

def load_synthea_data(data_directory: str) -> List[EHRDocument]:
    parser = SyntheaParser(data_directory)
    return parser.parse_data()

# Initialize the EHR Summarizer
config = OllamaConfig()
summarizer = EHRSummarizer(config, persist_dir="./ehr_chroma_db")

# Check if we need to create the index
if summarizer.index is None:
   # ehr_data = load_synthea_data('/path/to/synthea/output')  # Update this path
    ehr_data = load_synthea_data(r'C:\Users\niiOkaiBright\Desktop\pyprojects\ehr_rag\ehr_chroma_db')
    summarizer.create_index(ehr_data)

def generate_summary(patient_id: str, task: str) -> str:
    if not patient_id:
        return "Please enter a valid Patient ID."
    return summarizer.summarize(task, patient_id)

# Define the Gradio interface
with gr.Blocks(title="EHR Summarizer", theme=gr.themes.Soft()) as app:
    gr.Markdown("# EHR Summarizer")
    gr.Markdown("This application summarizes Electronic Health Records (EHR) based on your queries.")
    
    with gr.Row():
        with gr.Column(scale=1):
            patient_id = gr.Textbox(label="Patient ID", placeholder="Enter patient ID (e.g., P0001)")
        with gr.Column(scale=2):
            task = gr.Radio(
                [
                    "Summarize overall health status",
                    "List and describe chronic conditions",
                    "Summarize recent encounters and their reasons",
                    "List current medications with dosages and costs",
                    "Summarize recent procedures and their costs",
                    "Analyze healthcare expenses and coverage",
                    "Custom query"
                ],
                label="Select a task",
                value="Summarize overall health status"
            )
    
    custom_query = gr.Textbox(label="Custom Query", placeholder="Enter your custom query here", visible=False)
    
    summarize_btn = gr.Button("Generate Summary", variant="primary")
    
    output = gr.Textbox(label="Summary", lines=15, interactive=False)
    
    def update_custom_query_visibility(choice):
        return gr.update(visible=choice == "Custom query")
    
    task.change(update_custom_query_visibility, task, custom_query)
    
    def on_summarize_click(patient_id, task, custom_query):
        query_map = {
            "Summarize overall health status": "provide a comprehensive summary of the patient's overall health status, including major conditions, recent encounters, and key health indicators.",
            "List and describe chronic conditions": "list and describe all chronic conditions the patient has been diagnosed with, including dates of onset and current status.",
            "Summarize recent encounters and their reasons": "summarize the patient's most recent medical encounters, including dates, types, reasons for visit, and outcomes.",
            "List current medications with dosages and costs": "list all current medications prescribed to the patient, including dosages, start dates, and total costs.",
            "Summarize recent procedures and their costs": "summarize all medical procedures the patient has undergone, including dates, descriptions, reasons, and costs.",
            "Analyze healthcare expenses and coverage": "analyze the patient's healthcare expenses and coverage, providing insights into their overall healthcare costs and insurance utilization.",
            "Custom query": custom_query
        }
        
        query = query_map.get(task, "Invalid task selected.")
        return generate_summary(patient_id, query)
    
    summarize_btn.click(on_summarize_click, inputs=[patient_id, task, custom_query], outputs=output)

    gr.Markdown("## Note")
    gr.Markdown("This application uses synthetic data generated by Synthea. Patient IDs are in the format P0001, P0002, etc.")

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7860,  # Default Gradio port
        share=False,  # Disable sharing
        inbrowser=True,  # Automatically open in default browser
    )