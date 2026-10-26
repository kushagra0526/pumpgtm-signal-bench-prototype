# signalbench: which "ready to buy" signals are worth a message, and how many messages go out after a no?

PumpGTM's site says it finds "people who signal that they're ready to buy" (a like on X, a comment on LinkedIn, a company that just started facing the problem) and reaches out on LinkedIn, X and email; once someone replies on any channel the remaining steps are cancelled (https://pumpgtm.com/). Two measurable questions follow: how much better than a plain ICP list are signal-based picks when some people like and comment on everything, and how many messages still reach someone who has already said no. This is a synthetic benchmark with known buyers. It does not model LinkedIn's rules or any anti-automation measures, and I make no claim about them.

## The setup
`signalbench/sim.py`: 20,000 prospects, 3% buyers (1.5% in held-out); each has an ICP-fit flag (85% of buyers, 25% of others) and a baseline engagement rate (heavy engagers like and comment on everything); buyers engage with relevant posts 6x (likes) and 10x (comments) more than their baseline in dev. Positive reply: 25% of buyers, 1% of others; 4-5% say no or stop. Contact 1,000 per campaign:
- **any ICP match** (random from the ICP list),
- **ICP + any like or comment**,
- **scored, signals as observed:** logistic regression on an earlier labelled campaign (fit, like, comment),
- **scored, signals relative to own activity:** like and comment divided by the person's own activity level, plus the level itself.
Then, for the likes-and-comments list: messages still sent to people who have already replied no, under three sequencing rules (a 3-step sequence): channels run independently (2 more messages), stop on any reply (about 0.5 in-flight messages, my assumption), stop and suppress the person (0).

## Results (40 campaigns per split)
Dev: seeds 0-39. Held-out: seeds 1000-1039, fixed in git first: 1.5% buyers, 30% heavy engagers, buyers engage only 3x and 5x more.

| per 1,000 contacts (mean) | held-out buyers contacted | held-out positive replies | dev buyers contacted | dev positive replies |
|---|---|---|---|---|
| any ICP match | 50 | 23 | 96 | 32 |
| ICP + any like or comment | 79 | **29.5** | 171 | 50 |
| scored, signals as observed | **80** | 28 | 172 | **51** |
| scored, signals relative to own activity | 73 | 26 | 147 | 46 |

| held-out, messages sent to someone who already said no, per 1,000 contacts | |
|---|---|
| channels run independently | 79.7 |
| stop on any reply, all channels | 19.9 |
| stop on any reply + suppress the person | 0 |

- Signals lift the number of real buyers among 1,000 contacts by about 60% (50 to 79) and positive replies by about 28%, but over 90% of the people contacted are still not buyers (about 39 of 1,000 say no whichever list is used).
- Normalising for heavy engagers did not help; it was slightly worse than the plain signals in both splits (I expected the opposite). A regression on the raw signals matched the simple "like or comment" rule.
- Sequencing matters more for complaints than the scoring does: independent channels send about 80 messages per 1,000 contacts to people who said no.

## Read this before believing the table
Everything here is generated: how often buyers like or comment, the reply rates and the "in-flight messages" figure are my assumptions, and real buyer signals are far noisier. I did not look into LinkedIn's terms for automation or proxies; PumpGTM's page describes routing LinkedIn activity through residential proxies "so it looks like you", which is a platform-rules question I did not model or test. Nothing here uses PumpGTM data or systems.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m signalbench.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with PumpGTM.
