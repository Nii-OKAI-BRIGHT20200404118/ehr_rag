import nltk
from nltk.translate.bleu_score import sentence_bleu

# Example reference and hypothesis
reference = [['patient' 'P001' 'cough' 'for' 'two' 'weeks' 'elevated' 'white' 'blood' 'cell' 'count' 'Follow-up' 'visit' 'Cough' 'has' 'improved' 'Continue' 'current' 'treatment']]
hypothesis = ['Patient' 'P001' 'cough' 'for' 'two' 'weeks' 'elevated' 'white' 'blood' 'cell' 'count' 'Follow-up' 'visits' 'continues' 'to' 'be' 'prescribed']

# Calculate BLEU-1 (only 1-grams)
bleu_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))

print(f"BLEU-1 Score: {bleu_1}")

# from evaluate import load
# bleu = load("bleu")
# prediction  = ["i", "have", "thirty", "six", "years"]
# references = ["i", "am", "thirty", "six", "years" "old"]
# bleu.compute(prediction=prediction, references=references)


# from evaluate import load

# # Load the BLEU metric
# # bleu = load('bleu')
# sacrebleu = load('sacrebleu')

# # Correctly formatted prediction and references
# prediction = ['i have thirty six years']  # A list of sentences (predictions)
# references = [['i am thirty years old', 'i am thirty six']]  # A list of lists of reference sentences

# # Compute BLEU score
# # bleu_result = bleu.compute(predictions=prediction, references=references)
# sacrebleu_result = sacrebleu.compute(predictions=prediction, references=references)

# # Print BLEU score
# # print("BLEU Score:", bleu_result)
# print("BLEU Score:", sacrebleu_result)
      


# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# # Example: One generated summary and its reference summary
# generated_summary = "Patient P001 had a cough for two weeks that was treated with antibiotics in January. However, there were no further signs of the infection and the treatment was continued until February. the patient had a blood test elevated white blood cell count.".split()
# reference_summary = "Patient P001, visit on 2023-01-15 and complains of persistent cough for 2 weeks and antibiotics were prescribed, at the lab Blood test shows elevated white blood cell count. another follow-up visit shows that Cough has improved. Continue current treatment.".split()
# # Compute BLEU score
# smoothing_function = SmoothingFunction().method4
# score = corpus_bleu(reference_summary, generated_summary, smoothing_function=smoothing_function)
# #bleu_score = sentence_bleu(reference_summary, generated_summary)

# # Output BLEU score
# #print(f"BLEU Score: {bleu_score}")
# print(f"BLEU Score: {score}")

