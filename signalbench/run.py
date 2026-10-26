import json

import numpy as np

from . import sim

METHODS = ["any ICP match", "ICP + any like or comment", "scored, signals as observed", "scored, signals relative to own activity"]
POLICIES = ["channels run independently", "stop on any reply, all channels", "stop on any reply + suppress the person"]


def score(seeds, held, k=1000):
    acc = {m: {"buyers_contacted": [], "positive_replies": [], "negatives": []} for m in METHODS}
    after = {p: [] for p in POLICIES}
    for s in seeds:
        pool, hist = sim.make_pool(s, held), sim.make_pool(s + 500000, held)
        for m in METHODS:
            idx = sim.pick(m, pool, hist, k)
            pos, neg = sim.outcomes(pool, idx, pool["rng"])
            acc[m]["buyers_contacted"].append(float(pool["buyer"][idx].sum()))
            acc[m]["positive_replies"].append(float(pos.sum()))
            acc[m]["negatives"].append(float(neg.sum()))
            if m == METHODS[1]:
                for p in POLICIES:
                    after[p].append(sim.messages_after_no(neg, p))
    out = {m: {k2: round(float(np.mean(v)), 1) for k2, v in d.items()} for m, d in acc.items()}
    out["_messages_after_a_no_per_1000_contacts (signal rule)"] = {p: round(float(np.mean(v)), 1) for p, v in after.items()}
    return out


def main():
    res = {"dev": score(range(0, 40), False), "held_out": score(range(1000, 1040), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print("  ", m, d)


if __name__ == "__main__":
    main()
