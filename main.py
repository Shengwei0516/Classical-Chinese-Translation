import os
import json
import torch
import random
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate.utils import set_seed
from torch.optim import AdamW
from tqdm.auto import tqdm


def parse_args():

    parser = ArgumentParser(description="Finetune a Taiwan-LLaMa on a Classical Chinese Translation")
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A json file containing the training data."
    )
    parser.add_argument(
        "--valid_file",
        type=str, 
        default=None,
        help="A json file containing the validation data."
    )
    parser.add_argument(
        "--test_file",
        type=str, 
        default=None,
        help="A json file containing the testing data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=0, 
        help="Total number of training epochs to perform."
        )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
        )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=11151033, 
        help="A seed for reproducible training."
        )
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="prediction.json",
        help="Path to the output prediction file."
    )
    parser.add_argument(
        "--use_data",
        type=int,
        default=None,
        help="How much training data did you use?"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--target_modules",
        nargs='*',
        default=None,
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        help="Path to the saved PEFT checkpoint."
    )
    args = parser.parse_args()
    
    return args


def get_prompt(instruction: str, strategy: str=None) -> str:
    train_data = [
        {
            "id": "db63fb72-e211-4596-94a4-69617706f7ef",
            "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
            "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
        },
        {
            "id": "c45dc061-cba4-4ede-b0e5-1a8fd5dfdf6a",
            "instruction": "太祖親自擬文，然後嵌在上麵，是： 上天保佑人民，朕纔能夠統率安撫。\n這句話在中國古代怎麼說：",
            "output": "太祖親為文鈒之曰： 上天祐民，朕乃率撫。"
        },
        {
            "id": "f74e9e16-dfc5-4c69-a606-378d09c08571",
            "instruction": "文言文翻譯：\n是年，遷北院樞密副使，復陳陰害太子計，乙辛從之。",
            "output": "答案：同年，十三遷任北院樞密副使，又陳說暗中謀害太子的計策，乙辛從之。"
        },
        {
            "id": "1dc02e83-7f91-42e2-9ae5-5689f052afce",
            "instruction": "今聽其所執，以壽光公就第，位同保傅，在三司之右。\n翻譯成現代文：",
            "output": "如今聽從他的意願，以壽光公身份迴宅第，地位如同太保太傅，在三司之上。"
        },
        {
            "id": "05f3e696-b62c-4a3d-a74a-999f1e958cac",
            "instruction": "將下麵句子翻譯成現代文：\n繼踵鬍母，仲舒雲盛，因修《榖梁》，韆鞦最篤。",
            "output": "繼承跟踵鬍母生，董仲舒達到頂盛，因循《榖梁》，夏侯韆鞦最忠誠。"
        },
    ]
    if strategy == "Zero-Shot":
        return f"請完成以下的文言文翻譯、白話文翻譯。翻譯句子的意思。USER: {instruction} ASSISTANT:"
    elif strategy == "Few-Shot":
        return f"USER: {train_data[0]['instruction']} ASSISTANT: {train_data[0]['output']}" \
                f"USER: {train_data[1]['instruction']} ASSISTANT: {train_data[1]['output']}" \
                f"USER: {train_data[2]['instruction']} ASSISTANT: {train_data[2]['output']}" \
                f"USER: {train_data[3]['instruction']} ASSISTANT: {train_data[3]['output']}" \
                f"USER: {train_data[4]['instruction']} ASSISTANT: {train_data[4]['output']}" \
                f"USER: {instruction} ASSISTANT:"
    else:
        return instruction


def get_bnb_config(quantization: bool=False) -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            )
    return quantization_config


def perplexity(
    model, tokenizer, data, max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_function = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        loss += [loss_function(shift_logits.transpose(1, 2), shift_label)]
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls), "loss": float(np.mean(loss))}


def learning_curves(result_history: list, output_dir: str) -> None:

    import matplotlib.pyplot as plt
    ppl = result_history["ppl"]
    loss = result_history["loss"]
    x = range(1, len(ppl) + 1)

    plt.figure()
    plt.plot(x, ppl, label="Valid")
    plt.title(f"Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Perplexity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves_ppl")

    plt.figure()
    plt.plot(x, loss, label="Valid")
    plt.title(f"Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves_loss")


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        config=config,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config(args.quantization)
        )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)
    else:
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    model.to("cuda")

    if args.epochs > 0 and args.train_file:

        data = {}
        data["train"]  = json.load(open(args.train_file, "r", encoding="utf-8"))
        if args.valid_file:
            data["valid"] = json.load(open(args.valid_file, "r", encoding="utf-8"))

        if args.use_data:
            if args.use_data < len(data['train']):
                random_indices = random.sample(range(len(data['train'])), args.use_data)
                data['train'] = data['train'].select(random_indices)

        def preprocess_function(data):

            data_size = len(data)
            instructions = [x["instruction"] for x in data]
            outputs = [x["output"] for x in data]

            # Tokenize data
            tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
            tokenized_outputs = tokenizer(outputs, add_special_tokens=False)

            # Format data
            for i in range(data_size):
                instruction_input_ids = [tokenizer.bos_token_id] + \
                    tokenized_instructions["input_ids"][i]
                output_input_ids = tokenized_outputs["input_ids"][i] + \
                    [tokenizer.eos_token_id]
                tokenized_instructions["input_ids"][i] = instruction_input_ids + \
                    output_input_ids
                tokenized_instructions["attention_mask"][i] = [
                    1] * len(tokenized_instructions["input_ids"][i])

                tokenized_instructions["input_ids"][i] = torch.tensor(
                    tokenized_instructions["input_ids"][i][:args.max_length])
                tokenized_instructions["attention_mask"][i] = torch.tensor(
                    tokenized_instructions["attention_mask"][i][:args.max_length])

            return tokenized_instructions

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        loss_fct = torch.nn.CrossEntropyLoss()
        tokenized_instructions = preprocess_function(data["train"])

        history = {
                "ppl": [],
                "loss": [],
            }
        torch.cuda.empty_cache()
        for epoch in range(args.epochs):
            model.train()
            for i in tqdm(range(len(data["train"])), desc=f"Epoch {epoch+1}", unit_scale=True, colour="red"):
                input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
                attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
                label = input_ids
                
                out_logits = model(input_ids, attention_mask=attn_mask).logits
                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_label = label[..., 1:].contiguous()
                loss = loss_fct(shift_logits.transpose(1, 2), shift_label)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (i+1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if args.valid_file:
                model.eval()  
                valid_result = perplexity(model, tokenizer, data["valid"])
                print(f"Valid ppl: {valid_result['mean_perplexity']}, loss: {valid_result['loss']}")
                history["ppl"].append(valid_result['mean_perplexity'])
                history["loss"].append(valid_result['loss'])

        if args.output_dir:

            os.makedirs(args.output_dir, exist_ok=True)
            config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            model.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "params.json"), 'w') as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(args.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)

            learning_curves(history, args.output_dir)
            print(f"\nThe final model has been saved to {args.output_dir}")

    if args.test_file:

        test_data = json.load(open(args.test_file, "r", encoding="utf-8"))
        instructions = [get_prompt(x["instruction"], strategy=args.strategy) for x in test_data]
        model.eval()
        predictions = []
        torch.cuda.empty_cache()
        for i in tqdm(instructions, desc="Prediction", unit_scale=True, colour="red"):
            with torch.no_grad():
                input_ids = tokenizer(i, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    )
                generated_tokens = generated_tokens.cpu().numpy()
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_input = [inp.strip() for inp in decoded_input]
                decoded_preds = [pred.strip() for pred in decoded_preds]
                predictions += [decoded_preds[0][len(decoded_input[0]):]]
        outputs = [
            {"id": i["id"], "output": p} for i, p in zip(test_data, predictions)
        ]

        with open(args.prediction_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
        print(f"\nThe prediction results have been saved to {args.prediction_path}")


if __name__ == "__main__":
    main()
