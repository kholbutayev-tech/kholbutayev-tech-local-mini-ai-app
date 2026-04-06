import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct" -> any ai model
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer

def load_model(device):
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        )

    model = model.to(device)
    model.eval()
    return model

def generate_reply(user_message, tokenizer, model, device, history):
    history.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": reply})
    return reply

def chat_loop(tokenizer, model, device):
    history = []
    
    print(f"Kholbutayev-tech Mini Local Ai App \nModel: {MODEL_ID}")
    print(f"Device: {device}")
    print("Type 'exit' to quit, 'clear' to reset memory.\n")

    while True:
        user_message = input("You: ").strip()

        if not user_message:
            continue

        if user_message.lower() == "exit":
            print("Goodbye.")
            break

        if user_message.lower() == "clear":
            history = []
            print("Memory cleared.\n")
            continue

        try:
            reply = generate_reply(user_message, tokenizer, model, device, history)
            print(f"AI: {reply}\n")
        except RuntimeError as e:
            print(f"Runtime error: {e}\n")
            break

def main():
    device = get_device()
    print(f"Loading on: {device}")

    tokenizer = load_tokenizer()
    model = load_model(device)

    chat_loop(tokenizer, model, device)

if __name__ == "__main__":
    main()