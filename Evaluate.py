from utilities import load_islamTrust_mc1_dataset,parse_indices,evaluate_mc1_question
import torch
import tqdm
import os
import json
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from huggingface_hub import login



def get_args():
    parser = argparse.ArgumentParser(description="Evaluate IslamTrust MC1 benchmark")
    parser.add_argument('--indices', type=str, default=None, help='Indices to evaluate (comma separated or range)')
    parser.add_argument('--indicesMod', type=int, default=1, help='')
    parser.add_argument('--indicesRemainder', type=int, default=0, help='')
    parser.add_argument('--outDir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--token', type=str, required=True, help='Authentication token')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model name or path')
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Tokenizer name or path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--enable_quantization', type=bool, default=True, help='Loading quantized model or not')
    parser.add_argument('--bits', type=int, default=4, help='Quantization bits (4 or 8)')
    parser.add_argument('--data_directory', type=str, default="Abderraouf000/IslamTrust-benchmark", help='Path to the dataset CSV file')
    parser.add_argument('--language', type=str, default = "Arabic", help = "Benchmark language")
    
    return parser.parse_args()












def main():


    args = get_args();
    
    args.outDir = args.outDir + f"/{args.model_name}-{args.language}";

    os.makedirs(args.outDir, exist_ok=True)

    token = args.token



    login(token=token);

    model_name = args.model_name;
    
    tokenizer_name = args.tokenizer_name;
    
    # Load model and tokenizer
    if args.enable_quantization:
        if args.bits == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                     bnb_4bit_quant_type='nf4',
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_compute_type=torch.float16)   
            
        elif args.bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("Only 4 or 8 bits are supported for quantization.")
        
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=quantization_config,
                                                     device_map='auto',
                                                     trust_remote_code=True);
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto',trust_remote_code=True)   



    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name);



    model.eval();


    
    tokenizer.pad_token = tokenizer.eos_token;



    questions = load_islamTrust_mc1_dataset(data_directory=args.data_directory, language = args.language);
    

    category_number = {"Misconceptions":0,"General":0,"Islamically discouraged":0,"Extraordinary events":0,"Other faiths":0,"Different Islamic opinions":0};
    


    for i, item in enumerate(questions):
        category_number[item['Type']] += 1;



    indices = parse_indices(args.indices) if args.indices is not None else list(range(len(questions)));

    results = [];

    correct_predictions = 0;

    total_predictions = 0;

    result_category = {"Misconceptions":0,"General":0,"Islamically discouraged":0,"Extraordinary events":0,"Other faiths":0,"Different Islamic opinions":0};

    question_length = len(questions);


    with torch.no_grad():
        
        with torch.amp.autocast('cuda'):
            
            for idx in tqdm.tqdm(indices):
                if idx >= question_length:
                    continue
                    
                if idx % args.indicesMod != args.indicesRemainder:
                    continue
                if os.path.exists(os.path.join(args.outDir, f'{idx}.json')):
                    # Load existing result
                    with open(os.path.join(args.outDir, f'{idx}.json'), 'r') as f:
                        existing_result = json.load(f)
                        if existing_result.get('is_correct') is not None:
                            correct_predictions += existing_result['is_correct']
                            total_predictions += 1
                    continue
                
                question_data = questions[idx];
                                

                # Evaluate the question
                result = evaluate_mc1_question(
                    model, tokenizer, 
                    question_data['question'], 
                    question_data['choices'], 
                    question_data['correct_answer_index']
                );
                
                # Track accuracy
                correct_predictions += result['is_correct']
                total_predictions += 1;
                result_category[question_data['Type']] += float(result['is_correct'])

                
                # Save result
                save_dict = {
                    'question': question_data['question'],
                    'category':question_data['Type'],
                    'choices': question_data['choices'],
                    'correct_answer_index': int(question_data['correct_answer_index']),
                    'predicted_index': int(result['predicted_index']),
                    'predicted_answer': question_data['choices'][int(result['predicted_index'])],
                    'correct_answer': question_data['choices'][int(question_data['correct_answer_index'])],
                    'is_correct': bool(result['is_correct']),
                    'log_probs': [float(lp) for lp in result['log_probs']],
                    'accuracy_so_far': float(correct_predictions / total_predictions)
                }
                
                with open(os.path.join(args.outDir, f'{idx}.json'), 'w') as f:
                    json.dump(save_dict, f, indent=2)
                
                results.append(save_dict)
                
                # Print progress
                if total_predictions % 10 == 0:
                    print(f"Progress: {total_predictions} questions, Accuracy: {correct_predictions/total_predictions:.3f}")

    # Final accuracy calculation
    final_accuracy = float(correct_predictions / total_predictions) if total_predictions > 0 else 0;

    print(f"Final IslamTrust MC1 Accuracy: {final_accuracy:.3f} ({correct_predictions}/{total_predictions})");


    print('Number of samples in each category is : ',category_number)
    print('Per category results',result_category);




    
if __name__ == "__main__":
    main()