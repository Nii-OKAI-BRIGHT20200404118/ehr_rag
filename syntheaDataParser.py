import json
from typing import List, Dict
from pathlib import Path

def parse_synthea_data(data_directory: str) -> List[EHRDocument]:
    ehr_documents = []
    data_path = Path(data_directory)
    
    for patient_file in data_path.glob('*.json'):
        with open(patient_file, 'r') as f:
            patient_data = json.load(f)
        
        patient_id = patient_data['id']
        records = []
        
        # Parse encounters
        for encounter in patient_data.get('entry', []):
            if 'resource' in encounter and encounter['resource'].get('resourceType') == 'Encounter':
                records.append({
                    'date': encounter['resource'].get('period', {}).get('start'),
                    'type': 'visit',
                    'notes': encounter['resource'].get('reason', [{}])[0].get('text', 'No notes available')
                })
        
        # Parse conditions (diagnoses)
        for condition in patient_data.get('entry', []):
            if 'resource' in condition and condition['resource'].get('resourceType') == 'Condition':
                records.append({
                    'date': condition['resource'].get('onsetDateTime'),
                    'type': 'diagnosis',
                    'notes': condition['resource'].get('code', {}).get('text', 'No diagnosis available')
                })
        
        # Parse medications
        for medication in patient_data.get('entry', []):
            if 'resource' in medication and medication['resource'].get('resourceType') == 'MedicationRequest':
                records.append({
                    'date': medication['resource'].get('authoredOn'),
                    'type': 'medication',
                    'notes': medication['resource'].get('medicationCodeableConcept', {}).get('text', 'No medication info available')
                })
        
        ehr_documents.append(EHRDocument(patient_id, records))
    
    return ehr_documents

# Usage
ehr_data = parse_synthea_data('/path/to/synthea/output')
summarizer.create_index(ehr_data)