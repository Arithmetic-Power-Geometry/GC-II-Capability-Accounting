from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gcii.modular_agent import CHANNELS
from gcii.baselines import exact_shapley, walsh_anova, sobol_from_walsh, truncated_mobius_prediction
from gcii.separation import parity_truth, find_balanced_projection_complete, full_projections_up_to, robdd_node_count

RESULTS=ROOT/'results'; FIGS=ROOT/'figures'
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

def label(S): return '∅' if not S else ''.join(c for c in CHANNELS if c in S)

def values_from_frame(df,col):
    out={}
    for r in df.itertuples():
        mask=getattr(r,'mask')
        if isinstance(mask,float) and np.isnan(mask): mask=''
        out[frozenset(str(mask))]=float(getattr(r,col))
    return out

def baseline_prediction_audit():
    av=values_from_frame(pd.read_csv(RESULTS/'agent_factorial.csv'),'accuracy')
    dv=values_from_frame(pd.read_csv(RESULTS/'digits_factorial_summary.csv'),'accuracy_mean')
    full=frozenset(CHANNELS); rows=[]
    for name,vals in [('controlled_agent',av),('digits',dv)]:
        shap=exact_shapley(vals,CHANNELS); walsh=walsh_anova(vals,CHANNELS); sob=sobol_from_walsh(vals,CHANNELS)
        for method,order in [('singleton_additive',1),('pairwise_truncated',2),('third_order_truncated',3)]:
            pred=truncated_mobius_prediction(vals,CHANNELS,full,order)
            rows.append({'benchmark':name,'method':method,'prediction':pred,'observed':vals[full],'absolute_error':abs(vals[full]-pred)})
        rows.append({'benchmark':name,'method':'exact_full_factorial','prediction':vals[full],'observed':vals[full],'absolute_error':0.0})
        pd.DataFrame([{'channel':c,'shapley_value':shap[c],'first_order_sobol':sob.get(frozenset([c]),0.0)} for c in CHANNELS]).to_csv(RESULTS/f'{name}_scalar_attribution.csv',index=False)
        pd.DataFrame([{'interaction':label(T),'order':len(T),'walsh_coefficient':v,'sobol_share':sob.get(T,np.nan) if T else np.nan} for T,v in walsh.items()]).sort_values(['order','interaction']).to_csv(RESULTS/f'{name}_anova_sobol.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'heldout_full_prediction.csv',index=False)
    methods=['singleton_additive','pairwise_truncated','third_order_truncated']; x=np.arange(3); width=.35
    fig,ax=plt.subplots(figsize=(7,4.5))
    for j,name in enumerate(['controlled_agent','digits']):
        g=df[(df.benchmark==name)&df.method.isin(methods)].set_index('method').loc[methods]
        ax.bar(x+(j-.5)*width,g.absolute_error.to_numpy(),width,label=name)
    ax.set_xticks(x); ax.set_xticklabels(['order 1','order <=2','order <=3']); ax.set_xlabel('Retained interaction order'); ax.set_ylabel('Absolute error predicting held-out full system'); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS/'heldout_prediction.pdf'); fig.savefig(FIGS/'heldout_prediction.png',dpi=180); plt.close(fig)
    return df

def separation_audit():
    rows=[]
    for n in range(5,13):
        k=2; easy=parity_truth(n); hard,seed=find_balanced_projection_complete(n,k,seed0=20260908+n*100,tries=10000)
        er=robdd_node_count(easy,n); hr=robdd_node_count(hard,n)
        rows.append({'n':n,'k':k,'easy_support':sum(easy),'hard_support':sum(hard),'easy_full_k_projections':full_projections_up_to(easy,n,k),'hard_full_k_projections':full_projections_up_to(hard,n,k),'easy_obdd_nodes':er,'hard_obdd_nodes':hr,'obdd_ratio':hr/max(er,1),'hard_seed':seed})
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'whole_envelope_separation_audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.2)); ax.plot(df.n,df.easy_obdd_nodes,marker='o',label='parity envelope'); ax.plot(df.n,df.hard_obdd_nodes,marker='o',label='matched random envelope'); ax.set_yscale('log'); ax.set_xlabel('Envelope dimension n'); ax.set_ylabel('Exact ROBDD nonterminal nodes (fixed order)'); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS/'whole_envelope_obdd_separation.pdf'); fig.savefig(FIGS/'whole_envelope_obdd_separation.png',dpi=180); plt.close(fig)
    return df

def write_audit(pred,sep):
    dg=pred[(pred.benchmark=='digits') & (pred.method!='exact_full_factorial')].sort_values('absolute_error')
    singleton=pred[(pred.benchmark=='digits') & (pred.method=='singleton_additive')].iloc[0]
    best=sep.iloc[-1]
    text=f'''# GC-II extension audit\n\n- Scalar baselines: exact Shapley, two-level factorial/Walsh ANOVA coefficients, and Sobol-style variance shares computed from the same 16-condition scalar set functions.\n- Important interpretation: scalar baselines and scalar Mobius accounting receive the same aggregate information; the task-level EAS advantage comes from preserving the whole-envelope object, not from claiming a superior scalar decomposition formula.\n- Held-out fully augmented digits prediction: singleton absolute error = {singleton.absolute_error:.6f}; best retained-order error = {dg.iloc[0].absolute_error:.6f} ({dg.iloc[0].method}).\n- Finite exact representation audit at n={int(best.n)}: both envelopes are balanced and have all coordinate projections through order k={int(best.k)} full; parity uses {int(best.easy_obdd_nodes)} ROBDD nodes and the deterministic matched sample uses {int(best.hard_obdd_nodes)} nodes (ratio {best.obdd_ratio:.2f}).\n- The asymptotic whole-envelope separation theorem is a mathematical proof using Shannon's classical circuit-counting lower bound as an imported ingredient; the ROBDD experiment is only a finite exact representation audit.\n'''
    (RESULTS/'EXTENSION_AUDIT.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    p=baseline_prediction_audit(); s=separation_audit(); write_audit(p,s)
    print(json.dumps({'heldout_prediction':p.to_dict(orient='records'),'separation':s.to_dict(orient='records')},indent=2))
