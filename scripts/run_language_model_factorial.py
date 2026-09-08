from __future__ import annotations
import json, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import matplotlib.pyplot as plt
from gcii.modular_agent import CHANNELS
from gcii.mobius import powerset, mobius_decomposition, reconstruct_from_mobius
from gcii.baselines import exact_shapley, truncated_mobius_prediction

MODEL_ID='google/flan-t5-small'
RESULTS=ROOT/'results'; FIGS=ROOT/'figures'; RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
TASKS=[
 {'id':'math01','family':'mathematics','q':'What is 17 multiplied by 6?','answer':'102','expr':'17*6'},
 {'id':'math02','family':'mathematics','q':'What is 144 divided by 12?','answer':'12','expr':'144/12'},
 {'id':'math03','family':'mathematics','q':'What is 35 plus 47?','answer':'82','expr':'35+47'},
 {'id':'know01','family':'knowledge','q':'What is the capital of France?','answer':'paris','evidence':'France has Paris as its capital city.'},
 {'id':'know02','family':'knowledge','q':'Which planet is known as the Red Planet?','answer':'mars','evidence':'Mars is commonly called the Red Planet.'},
 {'id':'know03','family':'knowledge','q':'What gas do plants absorb from the atmosphere for photosynthesis?','answer':'carbon dioxide','evidence':'Plants absorb carbon dioxide during photosynthesis.'},
 {'id':'reason01','family':'reasoning','q':'All flims are zorks. No zorks are blue. Can any flim be blue? Answer yes or no.','answer':'no'},
 {'id':'reason02','family':'reasoning','q':'If Ana is older than Ben and Ben is older than Cara, is Ana older than Cara? Answer yes or no.','answer':'yes'},
 {'id':'reason03','family':'reasoning','q':'A box contains only red balls and blue balls. If a ball is not red, must it be blue? Answer yes or no.','answer':'yes'},
 {'id':'tool01','family':'tool-use','q':'Use arithmetic to evaluate 23 times 19.','answer':'437','expr':'23*19'},
 {'id':'tool02','family':'tool-use','q':'Evaluate 1000 minus 487.','answer':'513','expr':'1000-487'},
 {'id':'tool03','family':'tool-use','q':'Evaluate 81 divided by 9.','answer':'9','expr':'81/9'},
 {'id':'retr01','family':'retrieval','q':'What is the verification code for Project Aster?','answer':'7319','evidence':'Internal note: the verification code for Project Aster is 7319.'},
 {'id':'retr02','family':'retrieval','q':'Which city hosts the fictional Oriole Lab in this benchmark?','answer':'ranchi','evidence':'Benchmark record: Oriole Lab is located in Ranchi.'},
 {'id':'retr03','family':'retrieval','q':'What color token is assigned to synthetic record K-17?','answer':'violet','evidence':'Synthetic record K-17 is assigned the color token violet.'},
]

def normalize(x):
    x=x.strip().lower(); x=re.sub(r'[^a-z0-9.\- ]+',' ',x); x=' '.join(x.split())
    nums=re.findall(r'-?\d+(?:\.\d+)?',x)
    return nums[-1].rstrip('0').rstrip('.') if nums and re.fullmatch(r'[\s\w.\-]*',x) else x

def calculator(expr):
    if not expr or not re.fullmatch(r'[0-9+\-*/(). ]+',expr): return None
    try:
        val=eval(expr,{'__builtins__':{}},{})
        if isinstance(val,float) and val.is_integer(): val=int(val)
        return str(val)
    except Exception: return None

def prompt_for(task,S):
    parts=['Follow this control rule: use supplied evidence and tool observations when present; reason carefully; return only the final short answer.' if 'L' in S else 'Answer the question briefly.']
    if 'I' in S and task.get('evidence'): parts.append('Evidence: '+task['evidence'])
    if 'A' in S and task.get('expr'): parts.append('Calculator observation: '+calculator(task['expr']))
    parts.append('Question: '+task['q'])
    return '\n'.join(parts)

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.manual_seed(20260908); torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID); model.eval()
    rows=[]
    for S in powerset(CHANNELS):
        for task in TASKS:
            prompt=prompt_for(task,S); inputs=tok(prompt,return_tensors='pt',truncation=True,max_length=256)
            beams=4 if 'R' in S else 1; t0=time.perf_counter()
            with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=24,num_beams=beams,do_sample=False,early_stopping=True if beams>1 else False)
            elapsed=time.perf_counter()-t0; text=tok.decode(out[0],skip_special_tokens=True)
            pred=normalize(text); ans=normalize(task['answer']); ok=(pred==ans) or (ans in pred and len(ans)>=3)
            rows.append({'mask':''.join(sorted(S)),'channels':'∅' if not S else ''.join(c for c in CHANNELS if c in S),'n_channels':len(S),'task_id':task['id'],'family':task['family'],'correct':int(ok),'prediction':text,'answer':task['answer'],'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(out.shape[1]),'seconds':elapsed,'num_beams':beams,'tool_calls':int('A' in S and bool(task.get('expr'))),'evidence_chars':len(task.get('evidence','')) if 'I' in S else 0})
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'language_model_factorial_raw.csv',index=False)
    agg=df.groupby(['channels','mask','n_channels'],as_index=False).agg(accuracy=('correct','mean'),mean_seconds=('seconds','mean'),mean_input_tokens=('input_tokens','mean'),mean_output_tokens=('output_tokens','mean'),mean_tool_calls=('tool_calls','mean')).sort_values(['n_channels','mask']); agg.to_csv(RESULTS/'language_model_factorial_summary.csv',index=False)
    fam=df.groupby(['channels','mask','family'],as_index=False).agg(accuracy=('correct','mean')); fam.to_csv(RESULTS/'language_model_family_summary.csv',index=False)
    vals={frozenset('' if pd.isna(r.mask) else str(r.mask)):float(r.accuracy) for r in agg.itertuples()}; coeff=mobius_decomposition(vals,CHANNELS); full=frozenset(CHANNELS)
    cdf=pd.DataFrame([{'interaction':'∅' if not J else ''.join(c for c in CHANNELS if c in J),'order':len(J),'coefficient':v,'abs_coefficient':abs(v)} for J,v in coeff.items()]).sort_values(['order','interaction']); cdf.to_csv(RESULTS/'language_model_mobius_accuracy.csv',index=False)
    shap=exact_shapley(vals,CHANNELS); held=[]
    for order in [1,2,3]:
        p=truncated_mobius_prediction(vals,CHANNELS,full,order); held.append({'max_order':order,'prediction':p,'observed':vals[full],'absolute_error':abs(vals[full]-p)})
    pd.DataFrame(held).to_csv(RESULTS/'language_model_heldout_prediction.csv',index=False)
    summary={'model_id':MODEL_ID,'tasks':len(TASKS),'configurations':16,'evaluations':len(df),'baseline_accuracy':vals[frozenset()],'full_accuracy':vals[full],'max_reconstruction_error':max(abs(vals[S]-reconstruct_from_mobius(coeff,S)) for S in vals),'higher_order_abs_mass':sum(abs(v) for J,v in coeff.items() if len(J)>=2),'shapley':shap,'heldout_prediction':held}
    (RESULTS/'language_model_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(range(len(cdf)),cdf.coefficient.to_numpy()); ax.axhline(0,linewidth=.8); ax.set_xticks(range(len(cdf))); ax.set_xticklabels(cdf.interaction,rotation=70); ax.set_xlabel('Intervention subset'); ax.set_ylabel('Mobius contribution to accuracy'); fig.tight_layout(); fig.savefig(FIGS/'language_model_interactions.pdf'); fig.savefig(FIGS/'language_model_interactions.png',dpi=180); plt.close(fig)
    piv=fam.pivot(index='family',columns='channels',values='accuracy'); fig,ax=plt.subplots(figsize=(8,4.5)); im=ax.imshow(piv.to_numpy(),aspect='auto',vmin=0,vmax=1); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index); ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,rotation=70); ax.set_xlabel('System configuration'); ax.set_ylabel('Task family'); fig.colorbar(im,ax=ax,label='Accuracy'); fig.tight_layout(); fig.savefig(FIGS/'language_model_family_heatmap.pdf'); fig.savefig(FIGS/'language_model_family_heatmap.png',dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
