from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np





def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)

def parse_indices(indices):
    return list(_parse_indices_helper(indices))




def get_answer_logprob(model, tokenizer, question, answer,chat = False):
    if chat:
        return get_answer_logprob_chat_template(model, tokenizer, question, answer);
    else:
        return get_answer_logprob_no_chat(model, tokenizer, question, answer);
        


    


def get_answer_logprob_no_chat(model, tokenizer, question, answer):
    """Calculate log-probability of answer given question"""
    # Format the input - simple Q: A: format
    
    prompt = f"{question}{answer}"

    # messages = [
    # {"role": "user", "content": prompt},
    # ];
    
    # prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.\n\n### Instruction:\n{question}\n\n### Response: {answer}";
    
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs,)
        logits = outputs.logits
    
    # Calculate log-probabilities for the answer tokens
    # We need to identify which tokens correspond to the answer

    
    question_part = f"{question}";

    
    # question_part = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.\n\n### Instruction:\n{question}\n\n### Response: ";
    
    question_tokens = tokenizer(question_part, return_tensors="pt")
    question_length = question_tokens['input_ids'].shape[1]
    
    answer_start = question_length - 1 
    answer_tokens = inputs['input_ids'][0][answer_start:]
    
    
    log_probs = F.log_softmax(logits[0], dim=-1)
    
    answer_log_probs = []
    for i, token_id in enumerate(answer_tokens[1:]):  # Skip first token (it's the one we're predicting from)
        if i + answer_start < log_probs.shape[0]:
            token_log_prob = log_probs[i + answer_start, token_id].item()
            answer_log_probs.append(token_log_prob)
    
    return np.mean(answer_log_probs) if answer_log_probs else float('-inf');




def get_answer_logprob_chat_template(model, tokenizer, question, answer):
    
    """Calculate average log-probability of the answer given the question using chat template."""
    
    # Build chat-format message with the answer appended
    if not answer:
        return float('-inf')
        
    full_response = answer.strip()
    messages = [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": full_response},
    ]

    # Tokenize full prompt using chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    input_ids = inputs["input_ids"][0]
    
    # Get where the assistant's answer starts
    # Tokenize the user-only message to get its length
    user_only_messages = [{"role": "user", "content": question.strip()}]
    user_only_tokens = tokenizer.apply_chat_template(
        user_only_messages,
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )["input_ids"][0]

    answer_start = user_only_tokens.shape[0] 
    answer_token_ids = input_ids[answer_start:]

    # Get log-softmax over vocab
    log_probs = F.log_softmax(logits[0], dim=-1)

    
    # Collect log-probabilities for answer tokens
    token_log_probs = []
    for i in range(len(answer_token_ids) - 1):
        token_id = answer_token_ids[i + 1]
        log_prob = log_probs[answer_start + i, token_id].item()
        token_log_probs.append(log_prob)

    return np.mean(token_log_probs) if token_log_probs else float('-inf')




    

def evaluate_mc1_question(model, tokenizer, question, choices, correct_answer_index):
    """Evaluate a single MC1 question using log-probability scoring"""
    log_probs = []
    
    for choice in choices:
        log_prob = get_answer_logprob(model, tokenizer, question, choice, True)
        log_probs.append(log_prob)
    
    # Find the choice with maximum log-probability
    predicted_index = np.argmax(log_probs)
    is_correct = predicted_index == correct_answer_index
    
    return {
        'predicted_index': predicted_index,
        'correct_index': correct_answer_index,
        'is_correct': is_correct,
        'log_probs': log_probs,
        'choices': choices
    }





def load_islamTrust_mc1_dataset(data_directory):

    dataset = load_dataset("csv" ,data_files= data_directory)['train']
    questions = []
    for item in dataset:
        choices = []
        choices.extend([item['Choice1'],item['Choice2'],item['Choice3'],item['Choice4']]);
        if item['Question']:
            questions.append({
                'question': item['Question'],
                'choices': choices,
                'correct_answer_index': int(item['Answer'])-1, # Find the correct answer
                'Type':item['Type'],
            })
        
    print(f"Loaded {len(questions)} MC1 questions")
    return questions
