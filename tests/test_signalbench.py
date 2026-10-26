import numpy as np

from signalbench import sim


def test_pool_buyer_rate_and_reproducible():
    a, b = sim.make_pool(1), sim.make_pool(1)
    assert np.array_equal(a["buyer"], b["buyer"]) and 0.02 < a["buyer"].mean() < 0.04


def test_logistic_fit_recovers_direction_of_signals():
    p = sim.make_pool(2)
    w = sim.fit_logistic(sim.features(p, False), p["buyer"].astype(float))
    assert w[1] > 0 and w[2] > 0 and w[3] > 0


def test_signal_methods_beat_random_icp_contacts():
    r = []
    for s in range(6):
        pool, hist = sim.make_pool(s), sim.make_pool(s + 99)
        r.append([pool["buyer"][sim.pick(m, pool, hist, 500)].sum() for m in ("any ICP match", "scored, signals as observed")])
    r = np.array(r)
    assert r[:, 1].mean() > 1.5 * r[:, 0].mean()


def test_suppression_sends_nothing_after_a_no():
    neg = np.array([True, False, True])
    assert sim.messages_after_no(neg, "stop on any reply + suppress the person") == 0.0
    assert sim.messages_after_no(neg, "channels run independently") == 4.0
