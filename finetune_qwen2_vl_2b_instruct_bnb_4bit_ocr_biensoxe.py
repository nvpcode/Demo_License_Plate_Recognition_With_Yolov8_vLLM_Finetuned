# =========================
# 1. Import thư viện
# =========================
import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer


# =========================
# 2. Cấu hình chung
# =========================
MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
OUTPUT_DIR = "outputs"
SEED = 3407
MAX_STEPS = 50

INSTRUCTION = """
You are a world-class OCR expert specializing in recognizing all types of vehicle license plates
(cars, motorbikes, trucks, etc.) in any weather or lighting condition, including blurred, dirty,
or low-contrast images. Your recognition must be precise and avoid any confusion between
similar-looking characters (e.g., '0' and 'O', '1' and 'I', '8' and 'B').

Analyze the given image, which may contain one or multiple license plates.
For each license plate detected, extract and return ONLY its exact content,
using only the following valid characters: digits (0-9), uppercase letters (A-Z),
the hyphen (-), and the dot (.).

List each license plate you find on a separate line, with no extra words,
symbols, or explanations.
"""

# =========================
# 3. Load model
# =========================
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=SEED,
)


# =========================
# 4. Load và xử lý dữ liệu
# =========================
dataset = load_dataset("EZCon/taiwan-license-plate-recognition", split="train")
dataset = dataset.remove_columns(["xywhr", "is_electric_car"])
dataset = dataset.rename_column("license_number", "text")

def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text"]}
                ]
            }
        ]
    }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]


# =========================
# 5. Test trước khi train
# =========================
FastVisionModel.for_inference(model)

test_image = dataset[2]["image"]
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": test_image},
        {"type": "text", "text": INSTRUCTION}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(test_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

print("\nBefore training:")
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


# =========================
# 6. Huấn luyện
# =========================
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

trainer_stats = trainer.train()


# =========================
# 7. Test sau khi train
# =========================
FastVisionModel.for_inference(model)

print("\nAfter training:")
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# =========================
# 8. Lưu mô hình
# =========================
model.save_pretrained_merged("files_model/unsloth_finetune", tokenizer)