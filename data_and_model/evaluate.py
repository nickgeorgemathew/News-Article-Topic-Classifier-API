import argparse
from transformers import Autotokenizer,AutomodelForSequenceClassification,TextClassificationPipeline
from dataset import load_data
import numpy as np
from sklearn.metrics import classification_report,accuracy_score


def main(model_dir):
    dataset=load_data("ag_news",split="test")
    tokenizer=Autotokenizer.from_pretrained(model_dir)
    model=AutomodelForSequenceClassification.from_pretrained(model_dir)
    pipe=TextClassificationPipeline(model=model,tokenizer=tokenizer,return_all_score=True,device=-1)
    text=dataset["text"]
    labels=dataset["label"]
    preds=[]
    for t in text:
        out=pipe(t,trunication=True,max_length=256)
        scores=[c["score"] for c in out[0]]
        pred=int(np.argmax(scores))
        preds.append(pred)
    print("Accuracy:", accuracy_score(labels, preds))
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["World","Sports","Business","Sci/Tech"]))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",default="../models/distilbert-agnews-v1")
    args=parser.parse_args()
    model_dir=args.model_name
    main(model_dir=model_dir)