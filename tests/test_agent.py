from gcii.modular_agent import CHANNELS,generate_tasks,solve

def test_full_system_solves_all_generated_tasks():
    tasks=generate_tasks(per_requirement=3)
    full=frozenset(CHANNELS)
    assert all(solve(t,full)[0]==t.answer for t in tasks)

def test_missing_required_channel_abstains():
    tasks=generate_tasks(per_requirement=2)
    for t in tasks:
        if t.required:
            missing=next(iter(t.required))
            enabled=frozenset(set(t.required)-{missing})
            assert solve(t,enabled)[0] is None
