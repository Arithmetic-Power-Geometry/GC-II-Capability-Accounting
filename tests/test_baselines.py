from gcii.mobius import powerset
from gcii.baselines import exact_shapley, walsh_anova, sobol_from_walsh, truncated_mobius_prediction

C=("R","I","A","L")

def test_shapley_efficiency():
    v={S:0.2+0.03*len(S)+0.01*("R" in S and "I" in S) for S in powerset(C)}
    s=exact_shapley(v,C)
    assert abs(sum(s.values())-(v[frozenset(C)]-v[frozenset()]))<1e-12

def test_walsh_sobol_are_well_formed():
    v={S: float(len(S)**2) for S in powerset(C)}
    w=walsh_anova(v,C); sob=sobol_from_walsh(v,C)
    assert len(w)==16
    assert abs(sum(sob.values())-1.0)<1e-12

def test_truncated_prediction_does_not_need_full_endpoint_for_order3():
    v={S:0.1+0.01*len(S)+0.02*(len(S)>=2)+0.01*(len(S)>=3)+0.005*(len(S)==4) for S in powerset(C)}
    p=truncated_mobius_prediction(v,C,frozenset(C),3)
    v2=dict(v); v2[frozenset(C)] = 999.0
    p2=truncated_mobius_prediction(v2,C,frozenset(C),3)
    assert abs(p-p2)<1e-12
