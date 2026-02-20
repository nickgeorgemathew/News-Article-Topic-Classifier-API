import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

def preprocessing_function(examples,tokenizer,text_column="text"):
    return tokenizer(examples[text_column],truncation=True,max_length=256)

def compute_metrics_fn(pred):
    metric_acc=eval.load("accuracy")
    metric_f1=eval.load("f1")
    logits,label=pred
    preds=np.argmax(logits,axis=-1)
    acc=metric_acc.compute(predictions=preds,references=label)["accuracy"]
    f1_macro=metric_f1.compute(predictions=preds,references=label,average="macro")["f1"]
    return{"accuracy":acc,"f1_macro":f1_macro}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",default="distilbert-base-uncased")
    parser.add_argument("--output_dir",default="../models/distilbert-agnews-v1")
    parser.add_argument("--epoch",type=int,default=2)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--learning_rate",type=float,default=2e-5)
    parser.add_argument("--seed",type=int,default=42)

    args=parser.parse_args()

    raw_datasets=load_dataset("ag_news")

    tokenizer=AutoTokenizer.from_pretrained(args.model_name)

    tokenised_train=raw_datasets['train'].map(lambda x:preprocessing_function(x,tokenizer),batched=True)

    tokenised_test=raw_datasets['test'].map(lambda x:preprocessing_function(x,tokenizer),batched=True)


    model=AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=4)

    training_args=TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        push_to_hub=False
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_test,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn
        )
    
    trainer.train()

    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)

    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()