import gradio as gr
import json
import logging
from typing import List, Dict
from dataclasses import dataclass
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

    def summarize(self, task: str, patient_id: str) -> str:
        try:
            prompt = (f"Based on the electronic health records for patient {patient_id}, please {task} "
                      f"Provide a concise, well-structured response using complete sentences.")
            
            response = self.query_engine.query(prompt)
            return response.response if hasattr(response, 'response') else str(response)
        except Exception as e:
            logging.error(f"An error occurred during summarization: {e}")
            return f"I'm sorry, but an error occurred: {e}. Please try again or contact support for assistance."

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

# Initialize the EHR Summarizer
config = OllamaConfig()
summarizer = EHRSummarizer(config)
ehr_data = load_sample_ehr_data()
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
            patient_id = gr.Textbox(label="Patient ID", placeholder="Enter patient ID (e.g., P001)")
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

    gr.Markdown("## Security Note")
    gr.Markdown("This application runs locally and is not shared over the internet to protect patient data confidentiality.")
    gr.Markdown("## Disclaimer")
    gr.Markdown("This is a demo application using sample data. In a real-world scenario, it would be connected to a secure, comprehensive EHR database with proper authentication and authorization mechanisms.")

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7860,  # Default Gradio port
        share=False,  # Disable sharing
        inbrowser=True,  # Automatically open in default browser
    )