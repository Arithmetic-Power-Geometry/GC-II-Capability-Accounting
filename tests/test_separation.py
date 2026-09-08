from gcii.separation import parity_truth, find_balanced_projection_complete, full_projections_up_to, robdd_node_count


def test_matched_low_order_projection_pair_and_obdd_gap():
    n=8; k=2
    easy=parity_truth(n)
    hard,_=find_balanced_projection_complete(n,k,tries=2000)
    assert sum(easy)==sum(hard)==2**(n-1)
    assert full_projections_up_to(easy,n,k)
    assert full_projections_up_to(hard,n,k)
    assert robdd_node_count(hard,n) > robdd_node_count(easy,n)
