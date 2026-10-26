"""Which buying signals are worth contacting, and how many messages go out after someone says no? Known truth.

Prospects have a hidden buyer flag. Observable: ICP fit, whether they liked / commented on relevant posts, and how much they engage
with anything (heavy engagers like and comment on everything, buyers or not).
"""
import numpy as np


def make_pool(seed, held_out=False, n=20000):
    rng = np.random.default_rng(seed)
    pi = 0.015 if held_out else 0.03
    buyer = rng.random(n) < pi
    heavy = rng.random(n) < (0.30 if held_out else 0.15)            # engages with everything
    base = np.where(heavy, 0.12, 0.012)                              # per-prospect baseline like rate
    fit = rng.random(n) < np.where(buyer, 0.85, 0.25)
    lift_like, lift_com = (3.0, 5.0) if held_out else (6.0, 10.0)      # how much more a buyer engages with relevant posts
    like = rng.random(n) < np.clip(base * np.where(buyer, lift_like, 1.0), 0, 0.9)
    comment = rng.random(n) < np.clip(base * 0.4 * np.where(buyer, lift_com, 1.0), 0, 0.9)
    return {"buyer": buyer, "heavy": heavy, "base": base, "fit": fit, "like": like, "comment": comment, "rng": rng}


def features(p, normalise):
    x = np.column_stack([np.ones(len(p["fit"])), p["fit"], p["like"], p["comment"]]).astype(float)
    if normalise:   # signals relative to the person's own engagement level (observable as their general activity rate)
        act = p["base"] / p["base"].mean()
        x = np.column_stack([x[:, :2], x[:, 2] / act, x[:, 3] / act, np.log(act)])
    return x


def fit_logistic(x, y, iters=25, l2=1e-3):
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-x @ w))
        g = x.T @ (p - y) + l2 * w
        h = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        w -= np.linalg.solve(h, g)
    return w


def pick(method, p, hist, k):
    """hist = an earlier pool with known outcomes the scorers may train on."""
    n = len(p["fit"])
    rng = p["rng"]
    if method == "any ICP match":
        cand = np.where(p["fit"])[0]
        return rng.choice(cand, min(k, len(cand)), replace=False)
    if method == "ICP + any like or comment":
        cand = np.where(p["fit"] & (p["like"] | p["comment"]))[0]
        return cand[:k] if len(cand) >= k else np.concatenate([cand, rng.choice(np.where(p["fit"] & ~(p["like"] | p["comment"]))[0], k - len(cand), replace=False)])
    norm = method == "scored, signals relative to own activity"
    w = fit_logistic(features(hist, norm), hist["buyer"].astype(float))
    s = features(p, norm) @ w
    return np.argsort(-s)[:k]


def outcomes(p, idx, rng):
    """Positive reply from buyers, polite 'no / stop' from some non-buyers. Returns (positive, negative) boolean arrays over idx."""
    b = p["buyer"][idx]
    pos = rng.random(len(idx)) < np.where(b, 0.25, 0.01)
    neg = ~pos & (rng.random(len(idx)) < np.where(b, 0.05, 0.04))
    return pos, neg


def messages_after_no(neg, policy, steps=3):
    """Messages still sent to people who already replied 'no': unsynced per-channel sequences keep going on the other channels."""
    after = {"channels run independently": steps - 1, "stop on any reply, all channels": 0.5, "stop on any reply + suppress the person": 0.0}[policy]
    return float(neg.sum() * after)
