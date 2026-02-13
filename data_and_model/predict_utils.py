
from  transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassififcation

LABELS=["World","Sports","Business","Sci/Tech"]


class ModelWrapper:
    def __init__(self,model_dir:str):
        tokenizer=AutoTokenizer.from_pretrained(model_dir)
        model=AutoModelForSequenceClassififcation.from_pretrained(model_dir)
        self.pipe=pipeline(model=model,tokenizer=tokenizer,return_all_score=True,device=-1)
        self.model_version=model_dir
    
    
    
    
    def predict(self,text:str,top_k:int=3):
        out=self.pipe(text,trunication=True,max_length=256)
        scores=out[0]
        probs=[0.0]*len(scores)
        for d in scores:
            idx=int(d["label"].split("_")[-1])
            probs[idx]=d["scores"]
        ranked=sorted([{"label":LABELS[i],"scores":float(probs[i])} for i in range(len(probs))],key=lambda x:x["score"],reverse=True)
        return ranked[:top_k],probs




