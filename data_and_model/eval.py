import argparse
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TextClassificationPipeline
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score


def main(model_dir):
    dataset=load_dataset("ag_news",split="test")
    tokenizer=AutoTokenizer.from_pretrained(model_dir)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir)
    pipe=TextClassificationPipeline(model=model,tokenizer=tokenizer,return_all_score=True,device=-1)
    text=dataset["text"]
    labels=dataset["label"]
    preds=[]
    for t in text:
        out=pipe(t,trunication=True,max_length=256)
        out=dict(out[0])
        label=out['label']
        label=label.strip("_")[-1]
        preds.append(int(label)) 
    print("Accuracy:", accuracy_score(labels, preds))
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["World","Sports","Business","Sci/Tech"]))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",default="../distilbert/distilbert-agnews-v1")
    args=parser.parse_args()
    model_dir=args.model_name
    main(model_dir=model_dir)