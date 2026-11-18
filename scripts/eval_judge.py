"""Measure the moment judge on a labeled set: should-clip vs should-not-clip windows.

Runs the app's real judge prompt and score combination with your config's AI engine, and reports how many
moments end on the right side of the threshold. Use it before and after changing the prompt or the model.

    python scripts/eval_judge.py config.yaml            # threshold 7
    python scripts/eval_judge.py config.yaml --repeat 2 # average out sampling noise
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

# (should_clip, hype z, transcript, chat messages)
CASES: list[tuple[bool, float, str, list[str]]] = [
    # ---- should become clips
    (True, 4.2, "NO WAY NO WAY he just hit a one versus four clutch in overtime chat is going insane",
     ["W", "CLUTCH", "NO WAY", "INSANE", "KEKW"]),
    (True, 3.8, "wait wait wait five hundred dollars from mike oh my god I'm actually shaking thank you so much",
     ["W DONO", "POG", "LETS GO", "MIKE W"]),
    (True, 3.5, "okay I'm just gonna jump across here easy and oh no no no NO I fell into the lava again",
     ["LMAO", "KEKW", "OMEGALUL", "L", "again"]),
    (True, 3.9, "are you kidding me that is the third time this guy has spawn killed me I'm uninstalling this game",
     ["RAGE", "LMAO", "uninstall", "KEKW", "Madge"]),
    (True, 4.6, "AHHH what was that what was that I'm not going back in there no",
     ["JUMPSCARE", "LMAO", "monkaS", "AHHHH", "KEKW"]),
    (True, 3.3, "okay guys I've been waiting months to tell you this but I'm going to TwitchCon as a speaker",
     ["POG", "LETS GO", "HYPE", "congrats", "W"]),
    (True, 3.1, "watch this, flick, flick, flick, three headshots in one second did you see that",
     ["CRACKED", "W AIM", "NO WAY", "aimbot", "insane"]),
    # ---- should not
    (False, 2.2, "okay so I'm just gonna grab some water real quick be right back guys",
     ["o7", "brb", "ok", "o7"]),
    (False, 2.0, "and that's how you mine iron, you just keep going down to level sixteen and use a stone pickaxe",
     ["ok", "nice", "ty"]),
    (False, 2.4, "thanks to everyone who followed today, welcome raiders, hi sarah, hi tom, hi jessica",
     ["hi", "hello", "hey", "o/"]),
    (False, 2.1, "loading loading this map takes forever to load honestly let me check my settings",
     ["lol", "slow", "ok"]),
    (False, 2.3, "this stream is sponsored by NordVPN, use my code to get three months free on a two year plan",
     ["ad", "ok", "skip"]),
    (False, 2.2, "soldier report to the command post immediately, the enemy has breached the outer wall, hold the line",
     ["?", "ok"]),
    (False, 2.0, "can you guys hear me okay is my mic too loud let me know in chat one two one two",
     ["yes", "good", "too loud", "fine"]),
]


async def run(config: str, threshold: float, repeat: int) -> None:
    from streamclipper.config import load_settings

    s = load_settings(config)
    from streamclipper.detect import llama_server, llm_judge
    from streamclipper.detect.engine import cat_weight, combine

    cfg = s.detect
    right, results = 0, []
    for should, hype, text, chat in CASES:
        words = [(10 + k * 0.4, w) for k, w in enumerate(text.split())]
        msgs = [(12 + k, m) for k, m in enumerate(chat * 4)]
        finals = []
        for _ in range(repeat):
            t = time.time()
            j = await llm_judge.judge(10, 40, words, msgs, hype=hype, min_len=15, max_len=55,
                                      context="Ranked grind to Champion")
            finals.append((combine(j.score, hype, cfg.peak_z, cat_weight(j.category), cfg.hype_bonus_per_z,
                                   cfg.hype_bonus_max), j.score, time.time() - t, j.title))
        final = sum(f[0] for f in finals) / len(finals)
        ok = (final >= threshold) == should
        right += ok
        results.append((ok, should, final, finals[0][1], finals[0][2], text))
        print(f"[{'ok ' if ok else 'BAD'}] {'CLIP' if should else 'skip'}  final={final:4.1f}  llm={finals[0][1]:4.1f}  "
              f"{finals[0][2]:4.1f}s  {text[:58]}", flush=True)
    llama_server.stop()
    pos = [r for r in results if r[1]]
    neg = [r for r in results if not r[1]]
    print(f"\ncorrect: {right}/{len(CASES)}   clips kept: {sum(r[0] for r in pos)}/{len(pos)}   "
          f"dull rejected: {sum(r[0] for r in neg)}/{len(neg)}   "
          f"mean final clip={sum(r[2] for r in pos) / len(pos):.1f} skip={sum(r[2] for r in neg) / len(neg):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--threshold", type=float, default=7.0)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    asyncio.run(run(a.config, a.threshold, a.repeat))
