# llm-as-judge-theory
Understanding Existing Work in Public Domain 


# Pseudo Code According to Paper

```
SETUP:
  questions = sample of questions from your evaluation set
  n_responses = 5          # candidate responses per question
  n_replications = 100     # how many times to re-judge same input
  
═══════════════════════════════════════════════
STEP 1: Generate Candidate Responses
═══════════════════════════════════════════════

for each question in questions:
    responses[question] = []
    for i in 1..n_responses:
        response = LLM_generate(
            prompt = question,
            use_chain_of_thought = True
        )
        responses[question].append(response)

═══════════════════════════════════════════════
STEP 2: Run the Judge — Repeatedly
═══════════════════════════════════════════════

judgments = {}   # shape: [questions x n_replications]

for each question in questions:
    judgments[question] = []
    
    # Shuffle response order ONCE per question (hold it fixed across replications)
    shuffled_responses = shuffle(responses[question])  # labeled [A-E]
    
    for rep in 1..n_replications:
        verdict = LLM_judge(
            prompt = build_prompt(question, shuffled_responses),
            seed = rep,          # <-- only thing that changes
            temperature = 0.25   # or whatever, but keep constant
        )
        # verdict is one of [A, B, C, D, E]
        judgments[question].append(verdict)

═══════════════════════════════════════════════
STEP 3: Encode Judgments as Numbers
═══════════════════════════════════════════════

# Convert letter choices to binary indicator matrix
# For each question, you get a [n_replications x n_responses] matrix
# where each row is a one-hot vector of the judge's pick

for each question in questions:
    judgment_matrix[question] = one_hot_encode(judgments[question])
    # shape: [100 x 5]

═══════════════════════════════════════════════
STEP 4: Compute McDonald's Omega per Question
═══════════════════════════════════════════════

for each question in questions:
    X = judgment_matrix[question]   # [100 x 5]
    
    # Factor analysis: decompose variance into
    # common factor (true signal) vs. unique/error variance
    factor_loadings, error_variances = factor_analysis(X, n_factors=1)
    
    sum_loadings   = sum(factor_loadings)
    sum_error_var  = sum(error_variances)
    total_variance = variance(sum_of_all_columns(X))
    
    omega[question] = (sum_loadings² ) / total_variance
    # omega ranges from 0 to 1

═══════════════════════════════════════════════
STEP 5: Interpret
═══════════════════════════════════════════════

overall_omega = mean(omega across all questions)

if overall_omega >= 0.90:   print("Excellent reliability")
elif overall_omega >= 0.80: print("Good reliability")
elif overall_omega >= 0.70: print("Acceptable reliability")
else:                       print("Unreliable judge — do not trust single-shot verdicts")
```
