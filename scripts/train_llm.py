from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments

def main():

    print("loading dataset...")
    ds = load_dataset("EleutherAI/the_pile_deduplicated", split="train")

    print("loading tokenizer GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Tokenisation of dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_ds = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    print("loading pre-trained model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")


    print("config args...")
    training_args = TrainingArguments(
        output_dir="./models/dallm",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
    )


    print("Trainer installation...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )


    print("begginning training...")
    trainer.train()


    print("save model and tokenizer...")
    model.save_pretrained("./models/mon_modele_llm")
    tokenizer.save_pretrained("./models/mon_modele_llm")

if __name__ == "__main__":
    main()
