from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM_model:

    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.model_name = model_name

        print(f"Loading model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True
        )

        print("Model loaded.")
        print("Device:", self.model.device)


    def generate_search_query(self, prompt):
        instruction = (
            "Generate a search query for arXiv by extracting the keywords from the following question:\n"
            f"{prompt}\n\n"
            "Return the search query only. Do not provide an explanation."
        )

        messages = [
            {
                "role": "user",
                "content": instruction
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        print(
            "Input tokens:",
            model_inputs["input_ids"].shape[1]
        )

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False
        )

        output_ids = generated_ids[
            0,
            model_inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )

        return answer.strip()