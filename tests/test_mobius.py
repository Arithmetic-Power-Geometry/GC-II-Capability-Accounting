from gcii.mobius import powerset,mobius_decomposition,reconstruct_from_mobius,additive_prediction

CHANNELS=("R","I","A","L")

def test_exact_reconstruction_random_table():
    values={S: ((sum(ord(c) for c in S)*17 + len(S)**3 + 13) % 101)/100 for S in powerset(CHANNELS)}
    coeff=mobius_decomposition(values,CHANNELS)
    for S,v in values.items():
        assert abs(reconstruct_from_mobius(coeff,S)-v) < 1e-12

def test_additive_iff_no_higher_interactions_example():
    values={S: 0.2 + sum({"R":.1,"I":.2,"A":.05,"L":.03}[c] for c in S) for S in powerset(CHANNELS)}
    coeff=mobius_decomposition(values,CHANNELS)
    assert all(abs(v)<1e-12 for J,v in coeff.items() if len(J)>=2)
    assert abs(additive_prediction(values,CHANNELS)-values[frozenset(CHANNELS)])<1e-12

def test_endpoint_nonidentifiability_constructive():
    c=("R","I")
    v1={frozenset():0.0,frozenset(["R"]):0.0,frozenset(["I"]):0.0,frozenset(c):1.0}
    v2={frozenset():0.0,frozenset(["R"]):0.5,frozenset(["I"]):0.5,frozenset(c):1.0}
    assert v1[frozenset()]==v2[frozenset()] and v1[frozenset(c)]==v2[frozenset(c)]
    m1=mobius_decomposition(v1,c); m2=mobius_decomposition(v2,c)
    assert m1[frozenset(c)] != m2[frozenset(c)]

def test_worst_case_factorial_identifiability_constructive():
    channels=("R","I","A")
    Q=frozenset(("R","I"))
    v1={S:0.0 for S in powerset(channels)}
    v2=dict(v1); v2[Q]=1.0
    observed=[S for S in powerset(channels) if len(S)<=2 and S!=Q]
    assert all(v1[S]==v2[S] for S in observed)
    m1=mobius_decomposition(v1,channels); m2=mobius_decomposition(v2,channels)
    assert m1[Q] != m2[Q]

def test_equal_scalar_accuracy_can_hide_task_attribution():
    channels=("R","I")
    subsets=list(powerset(channels))
    A={frozenset():(0,0),frozenset(["R"]):(1,0),frozenset(["I"]):(0,1),frozenset(channels):(1,1)}
    B={frozenset():(0,0),frozenset(["R"]):(0,1),frozenset(["I"]):(1,0),frozenset(channels):(1,1)}
    accA={S:sum(A[S])/2 for S in subsets}; accB={S:sum(B[S])/2 for S in subsets}
    assert accA==accB
    t1A={S:A[S][0] for S in subsets}; t1B={S:B[S][0] for S in subsets}
    mA=mobius_decomposition(t1A,channels); mB=mobius_decomposition(t1B,channels)
    assert mA[frozenset(["R"])] != mB[frozenset(["R"])]
