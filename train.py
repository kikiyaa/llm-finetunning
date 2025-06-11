from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. 데이터셋 로드
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 토크나이저 및 모델
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# 3. 토큰화 함수
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# 4. 전체 토큰화 (batched=True + 병렬 처리)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. GPU용 학습 설정
training_args = TrainingArguments(
    output_dir="./gpt2-wikitext-finetuned",
    per_device_train_batch_size=4,      # VRAM에 따라 조절 (8 이상도 가능)
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,                          # ✅ GPU에서 속도/메모리 최적화
    push_to_hub=False,
    evaluation_strategy="epoch"         # epoch마다 검증
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 8. 학습 실행
trainer.train()

# 9. 저장
trainer.save_model("./gpt2-wikitext-finetuned")
tokenizer.save_pretrained("./gpt2-wikitext-finetuned")