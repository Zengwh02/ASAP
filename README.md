# ASAP
This is the official repository for the paper:

“Pruning the Unsurprising: Efficient Code Reasoning via First-Token Surprisal in Large Language Models.”

In this paper, we propose **ASAP** (**A**nchor-guided, **S**urpris**a**l-polished **P**runing), a novel two-stage framework for CoT compression.

---

## 📁 Repository Structure

This repository is organized into the following directories:
```
	eval/: Evaluation scripts to assess model performance on various code reasoning benchmarks.
	prune/: Code implementing the ASAP pruning algorithm.
	sft/: Scripts for full-parameter supervised fine-tuning (SFT) of models.
	utils/: Helper utilities and common functions used throughout the codebase.
```

---

## 🤖 Model Release

We release LogicCoder-7B and LogicCoder-8B, using our ASAP-pruned dataset derived from open-r1/codeforces-cots.

You can try our model on Hugging Face: [LogicCoder-7B](https://huggingface.co/azzzacs/LogicCoder-7B) and [LogicCoder-8B](https://huggingface.co/azzzacs/LogicCoder-8B)

---

## 🔧 Usage

We recommend **explicitly activating reasoning mode by inserting ```<think>``` in the prompt**.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("azzzacs/LogicCoder-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("azzzacs/LogicCoder-7B", device_map="auto", trust_remote_code=True).eval()
message = [{"role": "user", "content": "Please write a Python quick sort algorithm.\n"}]
prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False) + "<｜Assistant｜><think>\n"
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
outputs = model.generate(
    model_inputs.input_ids,
    max_new_tokens=4096,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0][len(model_inputs.input_ids[0]):], skip_special_tokens=False))
```

