# Gemma-4 Emotion Concepts Report

- Model: `mlx-community/gemma-4-e2b-it-4bit`
- Layer: `23` / `34`
- Emotions: happy, sad, angry, anxious, calm
- Train stories: `50`
- Eval stories: `20`
- Neutral texts: `10`
- Sampling: `temperature=1.0`, `top_k=64`, `top_p=0.95`, `min_p=None`, `repetition_penalty=None`, `presence_penalty=None`
- Neutral PCs projected out: `4`
- Held-out top-1 accuracy: `0.350`

## Dataset Design

- Matched train situations: `10` x `5` emotions = `50` stories
- Held-out evaluation situations: `4`

## Train Story Counts

- `happy`: 10 stories, 201.1 tokens on average
- `sad`: 10 stories, 213.6 tokens on average
- `angry`: 10 stories, 213.1 tokens on average
- `anxious`: 10 stories, 238.2 tokens on average
- `calm`: 10 stories, 220.4 tokens on average

## Held-Out Confusion

- `happy` -> happy:1, sad:0, angry:0, anxious:2, calm:1
- `sad` -> happy:0, sad:1, angry:0, anxious:2, calm:1
- `angry` -> happy:0, sad:1, angry:3, anxious:0, calm:0
- `anxious` -> happy:1, sad:0, angry:1, anxious:0, calm:2
- `calm` -> happy:0, sad:0, angry:1, anxious:1, calm:2

## Logit-Lens Tokens

- `happy` up: that, that, !, !,, toutefois, !'), optimizations, però
- `happy` down: 所谓, Proposed, অনেক, Proposed, many, Makes, Necessary, いわ
- `sad` up: 也不是, theirs, 也不, 但也, тоже, 也很, 上也, ไม่ใช่
- `sad` down: 르면, If, as, భాగంగా, tempted, quaisquer, пользу, 되면
- `angry` up: both, said, Specifically, ՝, mselves, Both, Said, told
- `angry` down: another, again, still, 也不, anymore, 또, 也不会, хороший
- `anxious` up: becomes, remains, seems, conviene, ?”,, stays, としては, merely
- `anxious` down: 력이, とその, ที่มี, 了他的, 及, }.\, ]., his
- `calm` up: سکے۔, ាន, конкре, ื้อ, ., einige, دوره, ための
- `calm` down: übrigens, **,, obstante, !,, by, Preference, ,</, embraced

## Steering Effects

### Prompt

Continue in 2-3 sentences: Mina held the envelope in both hands before sliding a finger under the seal.

- Baseline probe logits: happy:-20.88, sad:-23.25, angry:-15.69, anxious:-10.00, calm:-13.75
- happy intervention deltas: happy:-0.50, sad:-0.69, angry:-2.92, anxious:-4.88, calm:+1.34
- sad intervention deltas: happy:+2.59, sad:+1.97, angry:+3.03, anxious:+4.12, calm:+0.48
- angry intervention deltas: happy:-0.72, sad:-0.50, angry:-0.46, anxious:-0.46, calm:-0.92
- anxious intervention deltas: happy:+1.15, sad:+2.22, angry:-1.80, anxious:+2.71, calm:-3.64
- calm intervention deltas: happy:+2.84, sad:+0.53, angry:+1.12, anxious:-2.61, calm:+0.36

### Prompt

Continue in 2-3 sentences: Theo paused in the kitchen when he noticed the back door was already open.

- Baseline probe logits: happy:-18.25, sad:-19.75, angry:-17.75, anxious:-9.75, calm:-16.75
- happy intervention deltas: happy:-1.51, sad:-1.62, angry:-0.64, anxious:-4.09, calm:-0.66
- sad intervention deltas: happy:+3.47, sad:+4.90, angry:+4.20, anxious:+3.99, calm:-0.02
- angry intervention deltas: happy:-3.07, sad:-4.19, angry:-2.52, anxious:+0.18, calm:-4.48
- anxious intervention deltas: happy:-1.20, sad:-0.97, angry:-4.86, anxious:-3.36, calm:-3.77
- calm intervention deltas: happy:+3.76, sad:+0.53, angry:+5.07, anxious:+2.85, calm:+5.11

### Prompt

Continue in 2-3 sentences: Lena refreshed the results page and watched a new line of text appear.

- Baseline probe logits: happy:-19.88, sad:-21.12, angry:-17.75, anxious:-14.75, calm:-16.12
- happy intervention deltas: happy:+0.96, sad:-0.67, angry:-1.27, anxious:-1.43, calm:+1.25
- sad intervention deltas: happy:-0.80, sad:+0.60, angry:+1.22, anxious:-0.45, calm:-2.19
- angry intervention deltas: happy:-0.67, sad:-1.50, angry:-0.66, anxious:+0.42, calm:-1.28
- anxious intervention deltas: happy:-0.26, sad:+0.89, angry:+2.02, anxious:+2.12, calm:-2.04
- calm intervention deltas: happy:+0.64, sad:-0.36, angry:-0.80, anxious:-0.96, calm:+0.68

## Held-Out Story Topics

- `happy` on `The character sits in a parked car outside a courthouse while rereading a short text message.`
- `happy` on `When the character opens the mailbox, there is a single handwritten envelope with no return address.`
- `happy` on `The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.`
- `happy` on `At dawn, the character checks an online portal and sees that a long-awaited result is finally available.`
- `sad` on `The character sits in a parked car outside a courthouse while rereading a short text message.`
- `sad` on `When the character opens the mailbox, there is a single handwritten envelope with no return address.`
- `sad` on `The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.`
- `sad` on `At dawn, the character checks an online portal and sees that a long-awaited result is finally available.`
- `angry` on `The character sits in a parked car outside a courthouse while rereading a short text message.`
- `angry` on `When the character opens the mailbox, there is a single handwritten envelope with no return address.`
- `angry` on `The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.`
- `angry` on `At dawn, the character checks an online portal and sees that a long-awaited result is finally available.`
- `anxious` on `The character sits in a parked car outside a courthouse while rereading a short text message.`
- `anxious` on `When the character opens the mailbox, there is a single handwritten envelope with no return address.`
- `anxious` on `The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.`
- `anxious` on `At dawn, the character checks an online portal and sees that a long-awaited result is finally available.`
- `calm` on `The character sits in a parked car outside a courthouse while rereading a short text message.`
- `calm` on `When the character opens the mailbox, there is a single handwritten envelope with no return address.`
- `calm` on `The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.`
- `calm` on `At dawn, the character checks an online portal and sees that a long-awaited result is finally available.`
