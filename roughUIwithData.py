import gradio as gr
import logging
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import csv
import json
from datetime import datetime
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from httpx import Timeout

logging.basicConfig(level=logging.INFO)

@dataclass
class OllamaConfig:
    llm_model: str = "gemma2:2b"
    embedding_model: str = "nomic-embed-text:v1.5"
    api_base_url: str = "http://localhost:11434"
    chunk_size: int = 512
    chunk_overlap: int = 10
    timeout: float = 30.0

class EHRDocument:
    def __init__(self, patient_id: str, records: List[Dict]):
        self.patient_id = patient_id
        self.records = records

class SyntheaParser:
    def __init__(self, data_directory: str):
        self.data_dir = Path(r'C:\Users\niiOkaiBright\Desktop\pyprojects\ehr_rag\Data\syntheaNewData')
        self.id_mapping = {}
        self.load_id_mapping()

    def load_id_mapping(self):
        mapping_file = self.data_dir / 'id_mapping.json'
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.id_mapping = json.load(f)
            logging.info(f"Loaded ID mapping for {len(self.id_mapping)} patients.")
        else:
            logging.warning("id_mapping.json not found. Will use original IDs.")

    def get_patient_id(self, original_id: str) -> str:
        return self.id_mapping.get(original_id, original_id)

    def parse_data(self) -> List[EHRDocument]:
        ehr_documents = {}
        
        # Parse patients
        patients_file = next(self.data_dir.glob('*patients.csv'))
        with open(patients_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = self.get_patient_id(row['Id'])
                ehr_documents[row['Id']] = EHRDocument(patient_id, [])

        # Parse other files
        for file_type in ['encounters', 'conditions', 'medications', 'procedures']:
            file_path = next(self.data_dir.glob(f'*{file_type}.csv'))
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    patient_key = row.get('PATIENT') or row.get('Id')
                    if patient_key not in ehr_documents:
                        logging.warning(f"Patient {patient_key} not found in patients file. Skipping record.")
                        continue
                    
                    record_type = file_type[:-1]  # Remove 's' from the end
                    ehr_documents[patient_key].records.append({
                        'date': row.get('START') or row.get('DATE', 'Unknown Date'),
                        'type': record_type,
                        'notes': f"{record_type.capitalize()}: {row.get('DESCRIPTION', 'No description available')}"
                    })

        # Sort records for each patient by date
        for doc in ehr_documents.values():
            doc.records.sort(key=lambda x: x['date'])

        return list(ehr_documents.values())

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
        except Exception as e:
            logging.error(f"Failed to initialize EHRSummarizer: {e}")
            raise RuntimeError(f"Initialization error: {e}")

    def create_index(self, ehr_documents: List[EHRDocument]):
        try:
            documents = [
                Document(text=self.document_to_string(doc), id_=doc.patient_id)
                for doc in ehr_documents
            ]
            nodes = self.node_parser.get_nodes_from_documents(documents)
            if not nodes:
                raise ValueError("No nodes were created from the documents")
            self.index = VectorStoreIndex(nodes)
            self.query_engine = self.index.as_query_engine()
            logging.info("Index created successfully")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            raise RuntimeError(f"Error creating index: {e}")

    def document_to_string(self, doc: EHRDocument) -> str:
        records_str = "\n".join([f"{r['date']} - {r['type']}: {r['notes']}" for r in doc.records])
        return f"Patient ID: {doc.patient_id}\n\nMedical Records:\n{records_str}"

    def summarize(self, task: str, patient_id: str) -> str:
        try:
            prompt = (f"Based on the electronic health records for patient {patient_id}, please {task} "
                      f"Provide a concise, well-structured response using complete sentences.")
            
            response = self.query_engine.query(prompt)
            return response.response if hasattr(response, 'response') else str(response)
        except Exception as e:
            logging.error(f"An error occurred during summarization: {e}")
            return f"I'm sorry, but an error occurred: {e}. Please try again or contact support for assistance."

def load_synthea_data(data_directory: str) -> List[EHRDocument]:
    parser = SyntheaParser(data_directory)
    return parser.parse_data()

# Initialize the EHR Summarizer
config = OllamaConfig()
summarizer = EHRSummarizer(config)
ehr_data = load_synthea_data(r'C:\Users\niiOkaiBright\Desktop\pyprojects\ehr_rag\Data\syntheaNewData')  # Update this path
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
            patient_id = gr.Textbox(label="Patient ID", placeholder="Enter patient ID (e.g., P0001 or original UUID)")
        with gr.Column(scale=2):
            task = gr.Radio(
                ["Summarize medical history", "List prescribed medications", "Summarize laboratory results", "Custom query"],
                label="Select a task",
                value="Summarize medical history"
            )
    
    custom_query = gr.Textbox(label="Custom Query", placeholder="Enter your custom query here", visible=False)
    
    summarize_btn = gr.Button("Generate Summary", variant="primary")
    
    output = gr.Textbox(label="Summary", lines=10, interactive=False)
    
    def update_custom_query_visibility(choice):
        return gr.update(visible=choice == "Custom query")
    
    task.change(update_custom_query_visibility, task, custom_query)
    
    def on_summarize_click(patient_id, task, custom_query):
        if task == "Custom query":
            query = custom_query
        elif task == "Summarize medical history":
            query = "provide a concise summary of the patient's medical history."
        elif task == "List prescribed medications":
            query = "list all medications prescribed to the patient."
        elif task == "Summarize laboratory results":
            query = "provide a summary of all laboratory results for the patient."
        else:
            return "Invalid task selected."
        
        return generate_summary(patient_id, query)
    
    summarize_btn.click(on_summarize_click, inputs=[patient_id, task, custom_query], outputs=output)

    gr.Markdown("## Note")
    gr.Markdown("This application uses synthetic data generated by Synthea. Patient IDs can be in the format P0001, P0002, etc., or the original Synthea UUIDs.")

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7860,  # Default Gradio port
        share=False,  # Disable sharing
        inbrowser=True,  # Automatically open in default browser
    )