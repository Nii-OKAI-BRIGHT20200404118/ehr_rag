import pandas as pd
from typing import List, Dict
from datetime import datetime

class MimicIIIParser:
    def __init__(self, data_directory: str):
        self.data_dir = data_directory
        self.patients_df = pd.read_csv(f'{self.data_dir}/PATIENTS.csv')
        self.admissions_df = pd.read_csv(f'{self.data_dir}/ADMISSIONS.csv')
        self.diagnoses_df = pd.read_csv(f'{self.data_dir}/DIAGNOSES_ICD.csv')
        self.procedures_df = pd.read_csv(f'{self.data_dir}/PROCEDURES_ICD.csv')
        self.prescriptions_df = pd.read_csv(f'{self.data_dir}/PRESCRIPTIONS.csv')
        self.noteevents_df = pd.read_csv(f'{self.data_dir}/NOTEEVENTS.csv')
        
        # Load ICD9 codes for diagnosis and procedure descriptions
        self.d_icd_diagnoses = pd.read_csv(f'{self.data_dir}/D_ICD_DIAGNOSES.csv')
        self.d_icd_procedures = pd.read_csv(f'{self.data_dir}/D_ICD_PROCEDURES.csv')

    def parse_data(self, max_patients: int = None) -> List[EHRDocument]:
        ehr_documents = []
        
        for _, patient in self.patients_df.iterrows():
            if max_patients and len(ehr_documents) >= max_patients:
                break
            
            patient_id = patient['SUBJECT_ID']
            records = []
            
            # Add admissions
            patient_admissions = self.admissions_df[self.admissions_df['SUBJECT_ID'] == patient_id]
            for _, admission in patient_admissions.iterrows():
                records.append({
                    'date': admission['ADMITTIME'],
                    'type': 'admission',
                    'notes': f"Admitted for {admission['DIAGNOSIS']}. Discharge location: {admission['DISCHARGE_LOCATION']}"
                })
            
            # Add diagnoses
            patient_diagnoses = self.diagnoses_df[self.diagnoses_df['SUBJECT_ID'] == patient_id]
            for _, diagnosis in patient_diagnoses.iterrows():
                icd9_code = diagnosis['ICD9_CODE']
                diagnosis_desc = self.d_icd_diagnoses[self.d_icd_diagnoses['ICD9_CODE'] == icd9_code]['SHORT_TITLE'].values
                diagnosis_desc = diagnosis_desc[0] if len(diagnosis_desc) > 0 else 'Unknown diagnosis'
                records.append({
                    'date': patient_admissions[patient_admissions['HADM_ID'] == diagnosis['HADM_ID']]['ADMITTIME'].values[0],
                    'type': 'diagnosis',
                    'notes': f"Diagnosed with {diagnosis_desc} (ICD9: {icd9_code})"
                })
            
            # Add procedures
            patient_procedures = self.procedures_df[self.procedures_df['SUBJECT_ID'] == patient_id]
            for _, procedure in patient_procedures.iterrows():
                icd9_code = procedure['ICD9_CODE']
                procedure_desc = self.d_icd_procedures[self.d_icd_procedures['ICD9_CODE'] == icd9_code]['SHORT_TITLE'].values
                procedure_desc = procedure_desc[0] if len(procedure_desc) > 0 else 'Unknown procedure'
                records.append({
                    'date': patient_admissions[patient_admissions['HADM_ID'] == procedure['HADM_ID']]['ADMITTIME'].values[0],
                    'type': 'procedure',
                    'notes': f"Underwent procedure: {procedure_desc} (ICD9: {icd9_code})"
                })
            
            # Add prescriptions
            patient_prescriptions = self.prescriptions_df[self.prescriptions_df['SUBJECT_ID'] == patient_id]
            for _, prescription in patient_prescriptions.iterrows():
                records.append({
                    'date': prescription['STARTDATE'],
                    'type': 'medication',
                    'notes': f"Prescribed {prescription['DRUG']} ({prescription['DOSE_VAL_RX']} {prescription['DOSE_UNIT_RX']})"
                })
            
            # Add notes (limiting to 5 most recent to manage size)
            patient_notes = self.noteevents_df[self.noteevents_df['SUBJECT_ID'] == patient_id].sort_values('CHARTTIME', ascending=False).head(5)
            for _, note in patient_notes.iterrows():
                records.append({
                    'date': note['CHARTTIME'],
                    'type': 'note',
                    'notes': f"{note['CATEGORY']} note: {note['TEXT'][:500]}..." # Truncating long notes
                })
            
            # Sort records by date
            records = sorted(records, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))
            
            ehr_documents.append(EHRDocument(str(patient_id), records))
        
        return ehr_documents

# Usage
parser = MimicIIIParser('/path/to/mimic-iii/data')
ehr_data = parser.parse_data(max_patients=100)  # Parse data for 100 patients
summarizer.create_index(ehr_data)