import json
import logging
from typing import List, Dict
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import gradio as gr

logging.basicConfig(level=logging.INFO)

@dataclass
class OllamaConfig:
    llm_model: str = "gemma2"
    embedding_model: str = "nomic-embed-text:v1.5"
    api_base_url: str = "http://localhost:11434/"
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

# Initialize the summarizer and load data
config = OllamaConfig()
summarizer = EHRSummarizer(config)
ehr_data = load_sample_ehr_data()
summarizer.create_index(ehr_data)

# Define the Gradio interface functions
def summarize_patient_history(patient_id: str):
    return summarizer.summarize("Provide a concise summary of the patient's medical history.", patient_id)

def list_medications(patient_id: str):
    return summarizer.summarize("List all medications prescribed to the patient.", patient_id)

def summarize_lab_results(patient_id: str):
    return summarizer.summarize("Summarize all lab results for the patient.", patient_id)

def custom_query(patient_id: str, task: str):
    return summarizer.summarize(task, patient_id)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### EHR Summarization System")
    
    with gr.Tab("Summarize Patient History"):
        patient_id_1 = gr.Textbox(label="Enter Patient ID")
        summary_button_1 = gr.Button("Summarize")
        output_1 = gr.Textbox(label="Summary")
        summary_button_1.click(summarize_patient_history, inputs=patient_id_1, outputs=output_1)
        
    with gr.Tab("List Medications"):
        patient_id_2 = gr.Textbox(label="Enter Patient ID")
        summary_button_2 = gr.Button("List Medications")
        output_2 = gr.Textbox(label="Medications")
        summary_button_2.click(list_medications, inputs=patient_id_2, outputs=output_2)
        
    with gr.Tab("Summarize Lab Results"):
        patient_id_3 = gr.Textbox(label="Enter Patient ID")
        summary_button_3 = gr.Button("Summarize Lab Results")
        output_3 = gr.Textbox(label="Lab Results Summary")
        summary_button_3.click(summarize_lab_results, inputs=patient_id_3, outputs=output_3)
        
    with gr.Tab("Custom Query"):
        patient_id_4 = gr.Textbox(label="Enter Patient ID (optional)")
        task_input = gr.Textbox(label="Enter Your Custom Query")
        summary_button_4 = gr.Button("Submit Query")
        output_4 = gr.Textbox(label="Response")
        summary_button_4.click(custom_query, inputs=[patient_id_4, task_input], outputs=output_4)

# Run the Gradio app
demo.launch()
