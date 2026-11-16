"""Integration · governance_check(): ethics gate (C) + conflict-of-interest gate (D)

Generated from the notebook (cells 33). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# INTEGRATION · governance_check(): ethics-training gate (C) + conflict-of-interest gate (D)   [notebook cell 33]
# ----------------------------------------------------------------------------------------------------
def decisions_for_option(option):
    """Turn an option under review into Section D `Decision`s (its outside parties are what gets screened)."""
    parties = ", ".join(option.get("parties", []))
    return [Decision(bill=option["label"], subject=f"{option['type'].replace('_', ' ')} for transit bus service; parties: {parties}",
                     action="approve operational action")]


def decisions_for_proposal(proposal):
    changes = "; ".join(f"{c['path']} {c['old']}→{c['new']}" for c in proposal["changes"])
    return [Decision(bill=proposal["title"], subject=f"transit scoring policy change: {changes}", action="approve rule change")]


def governance_check(name, decision, option=None, proposal=None, policy=None):
    """Return (problems, notes). Problems block the submission; notes are logged in the audit record."""
    policy = policy or APP.PM.current
    g = policy.get("governance_gates", {})
    problems, notes = [], []
    if not g.get("enabled", True) or not name.strip():
        return problems, notes
    # ---- Section C: ethics training
    st = eth_check_official(name)
    if "error" in st:
        msg = f"{name} is not on the ethics-training roster"
        (problems if g.get("unknown_person") == "block" else notes).append(msg)
    elif st["status"] in g.get("training_block_statuses", []) and decision in g.get("training_gate_decisions", []):
        why = f"missing {', '.join(st['missing_courses'])}" if st["missing_courses"] else (
            "refresher overdue" if st["days_until_refresher_due"] is not None else "never trained")
        problems.append(f"{name} is NON_COMPLIANT on ethics training ({why}) and cannot {decision.lower()} — another reviewer must act")
    elif st["status"] == "AT_RISK":
        notes.append(f"{name}: ethics refresher due in {st['days_until_refresher_due']} days")
    else:
        notes.append(f"{name}: ethics training {st['status']}")
    # ---- Section D: conflict of interest
    if decision in g.get("coi_gate_decisions", []) and (option or proposal):
        disc = STAFF_DISCLOSURES.get(name.strip())
        if disc is None:
            notes.append(f"{name}: no conflict-of-interest disclosure on file")
        else:
            decs = decisions_for_option(option) if option else decisions_for_proposal(proposal)
            screen = rule_screen(disc["facts"], decs, g.get("coi_threshold", COI_THRESHOLD))
            name_hits = [h for h in screen["hits"] if h["match"] == "name" and h["score"] >= screen["threshold"]]
            if name_hits:
                h = name_hits[0]
                problems.append(f"possible conflict of interest: {name}'s {h['owner']} holding '{h['holding']}' ({h['kind']}) is a party "
                                f"to this option (score {h['score']} ≥ {screen['threshold']}) — recuse; another reviewer must approve")
            elif screen["hits"]:
                notes.append(f"{name}: conflict screen hits below the recusal rule — " +
                             "; ".join(f"{h['holding']} ({h['match']}, {h['score']})" for h in screen["hits"]))
            else:
                notes.append(f"{name}: conflict screen clear")
    return problems, notes


print("✓ Governance gate ready (training: Section C rules · independence: Section D rule screener)")
