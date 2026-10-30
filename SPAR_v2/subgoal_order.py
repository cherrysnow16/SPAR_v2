"""Most-recent-subgoal beliefs for the GraphGPO backend (SPAR v2).

Per training batch, one LLM call:
  1. each trajectory proposes up to 5 subgoals with achievement steps (self prompt, JSON schema);
     steps refer to the cleaned action list and are mapped back to original step indices
  2. labels (algorithm.spar.subgoal_label):
       text   - subgoal strings of a uid-group are merged by LSH (MinHash + exact Jaccard check + union-find)
       action - label derived from the action at the assigned step ("heat mug 1 with microwave 1" -> "heat mug");
                navigation/inspection steps get no label, the k-th distinct instance of a label gets "#k"
       action_lsh - (WebShop) the action at the assigned step with its argument kept: search queries of a
                uid-group are merged by LSH ("search: <canonical query>"), product/option clicks match exactly
                ("item: b07...", "option: large"), click[buy now] -> "buy"; page navigation gets no label
  3. belief at state j = merged label of the trajectory's subgoal achieved most recently before j
     (action o achieved it -> visible from state o+1)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from SPAR_v2.belief_pipeline import _dump_subgoal_records, run_llm_batch
from SPAR_v2.subgoal_lsh import cluster_subgoals
from SPAR_v2.subgoal_variants import SCHEMAS, self_prompt


def _parse_self(resp: str, n_actions: int):
    try:
        items = json.loads(resp)["subgoals"]
    except Exception:
        return None
    out = []
    for x in items:
        text, st = str(x.get("subgoal", "")).strip(), x.get("step")
        if text:
            out.append((text, st if isinstance(st, int) and 0 <= st < n_actions else None))
    return out


# (action pattern, label format over the groups without instance numbers, group holding the object instance)
_ACTION_LABELS = (
    (r"take (.+?) from .+", "take {0}", 1),
    (r"move (.+?) to (.+)", "put {0} {1}", 1),
    (r"(heat|cool|clean) (.+?) with .+", "{0} {1}", 2),
    (r"use (.+)", "use {0}", 1),
)


def action_label(action: str):
    """take/put/heat/cool/clean/use action -> (label without instance numbers, object instance); else None."""
    a = action.lower().strip()
    for pat, fmt, obj in _ACTION_LABELS:
        m = re.fullmatch(pat, a)
        if m:
            return fmt.format(*(re.sub(r"\s+\d+$", "", g) for g in m.groups())), m.group(obj)
    return None


_WEBSHOP_NAV = frozenset({"back to search", "next >", "< prev", "description", "features", "reviews"})


def webshop_action_kind(action: str):
    """search/click action -> (kind, argument) with kind in search/item/option/buy; navigation or unparsable -> None."""
    m = re.fullmatch(r"(search|click)\[(.+)\]", action.lower().strip())
    if not m:
        return None
    verb, arg = m.group(1), m.group(2).strip()
    if verb == "search":
        return "search", arg
    if arg in _WEBSHOP_NAV:
        return None
    if arg == "buy now":
        return "buy", ""
    is_asin = re.fullmatch(r"[a-z0-9]{10}", arg) is not None and any(c.isdigit() for c in arg)
    return ("item" if is_asin else "option"), arg


def _action_lsh_events(group_own: dict, group_actions: dict, threshold: float) -> dict:
    """uid-group {traj: own} -> {traj: [(cleaned step, label)]}; search queries are merged by LSH across the group."""
    kinds = {i: [(st, webshop_action_kind(group_actions[i][st]))
                 for st in sorted({st for _, st in own or [] if st is not None})]
             for i, own in group_own.items()}
    queries = [k[1] for evs in kinds.values() for _, k in evs if k and k[0] == "search"]
    canon = cluster_subgoals(queries, threshold=threshold) if queries else {}

    def label(kind, arg):
        return {"search": f"search: {canon.get(arg, arg)}", "buy": "buy"}.get(kind, f"{kind}: {arg}")

    return {i: [(st, label(*k)) for st, k in evs if k] for i, evs in kinds.items()}


def _action_events(own, actions: list[str]) -> list[tuple[int, str]]:
    """(subgoal text, cleaned step) list -> [(cleaned step, action label)] in step order."""
    seen = defaultdict(list)
    out = []
    for st in sorted({st for _, st in own or [] if st is not None}):
        lab = action_label(actions[st])
        if lab is None:
            continue
        label, inst = lab
        if inst not in seen[label]:
            seen[label].append(inst)
        k = seen[label].index(inst) + 1
        out.append((st, label if k == 1 else f"{label} #{k}"))
    return out


def annotate_order_recent(trajs: list[dict], tokenizer, actor_rollout_wg, config) -> list[list[frozenset]]:
    """trajs[i]: {"group", "task", "won", "actions" (cleaned), "orig_idx" (cleaned idx -> original step), "n_states"}
    -> per trajectory, per original state position: frozenset() or frozenset({merged subgoal label})"""
    beliefs = [[frozenset()] * t["n_states"] for t in trajs]
    if not trajs:
        return beliefs
    env = "webshop" if "webshop" in config.env.env_name.lower() else "alfworld"
    prompts = [self_prompt(t["task"], t["actions"], t["won"], "json_constrained", env) for t in trajs]
    responses = run_llm_batch(prompts, tokenizer, actor_rollout_wg, config,
                              max_tokens=int(config.algorithm.get("spar_belief_max_tokens", 2048)),
                              json_schema=SCHEMAS["self"])
    own = [_parse_self(r, len(t["actions"])) for r, t in zip(responses, trajs)]
    _dump_subgoal_records([{"mode": "order_recent", "group": t["group"], "won": t["won"],
                            "n_actions": len(t["actions"]), "resp_tokens": len(tokenizer.encode(r)),
                            "parsed": o is not None, "response": r}
                           for t, r, o in zip(trajs, responses, own)])

    by_group = defaultdict(list)
    for i, t in enumerate(trajs):
        by_group[t["group"]].append(i)
    thr = float(config.algorithm.spar.get("lsh_threshold", 0.6))
    label_mode = config.algorithm.spar.get("subgoal_label", "text")
    stats = Counter(parse_fail=sum(o is None for o in own))
    for idx in by_group.values():
        texts = [s for i in idx if own[i] for s, _ in own[i]]
        if not texts:
            continue
        if label_mode == "action":
            labeled = {i: _action_events(own[i], trajs[i]["actions"]) for i in idx}
        elif label_mode == "action_lsh":
            labeled = _action_lsh_events({i: own[i] for i in idx}, {i: trajs[i]["actions"] for i in idx}, thr)
        else:
            canon = cluster_subgoals(texts, threshold=thr)
            labeled = {i: [(st, canon[s]) for s, st in own[i] or [] if st is not None] for i in idx}
        stats["labels"] += len({lab for evs in labeled.values() for _, lab in evs})
        for i in idx:
            events = sorted((trajs[i]["orig_idx"][st], lab) for st, lab in labeled[i])
            cur, e, row = frozenset(), 0, []
            for j in range(trajs[i]["n_states"]):
                while e < len(events) and events[e][0] < j:
                    cur = frozenset({events[e][1]})
                    e += 1
                row.append(cur)
            beliefs[i] = row
            stats["trajs_with_subgoal"] += bool(events)
    stats["trajs"] = len(trajs)
    print(f"[SPAR v2 recent] label={label_mode} {dict(stats)}")
    return beliefs
