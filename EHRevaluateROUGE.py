# from rouge_score import rouge_scorer

# # Example: One generated summary and its reference summary
# generated_summary = "'Patient' 'P001' 'cough' 'for' 'two' 'weeks' 'elevated' 'white' 'blood' 'cell' 'count' 'Follow-up' 'visits'"
# reference_summary = "'patient' 'P001' 'cough' 'for' 'two' 'weeks' 'elevated' 'white' 'blood' 'cell' 'count' 'Follow-up' 'visit"

# # Initialize the ROUGE scorer
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# scores = scorer.score(reference_summary, generated_summary)

# # Output the ROUGE scores
# print(scores)

from rouge_score import rouge_scorer


# Example: One generated summary and its reference summary
generated_summary = "Patient P001 had a cough for two weeks that was treated with antibiotics in January. However, there were no further signs of the infection and the treatment was continued until February. In February, the patient had a blood test which revealed an elevated white blood cell count, but this was not significant or related to any previous medical history."
reference_summary = "Patient P001, visit on 2023-01-15 and complains of persistent cough for 2 weeks and antibiotics were prescribed, at the lab Blood test shows elevated white blood cell count. another follow-up visit shows that Cough has improved. Continue current treatment."

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)

# Output the ROUGE scores
print(scores)
