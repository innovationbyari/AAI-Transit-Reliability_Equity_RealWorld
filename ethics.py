"""Section C · Ethics training and compliance monitor (schemas, tools, agent, rosters)

Generated from the notebook (cells 22, 23, 24, 25). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# C · Structured outputs (schemas)   [notebook cell 22]
# ----------------------------------------------------------------------------------------------------
class Official(BaseModel):
    name: str
    department: str = ""
    completed: List[str] = []
    last_training: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD, or null if never trained")

    @field_validator("last_training")
    @classmethod
    def _iso_date(cls, v):                               # reject malformed dates with a clear message
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"last_training must be YYYY-MM-DD, got {v!r}")
        return v


class ActionItem(BaseModel):
    official: str
    status: Literal["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]
    missing_courses: List[str] = []
    action: str
    priority: Literal["high", "medium", "low"]


class ComplianceReport(BaseModel):                  # the agent's final answer
    answer: str = Field(description="Direct answer to the user's question, max 80 words")
    compliance_rate_pct: float = Field(ge=0, le=100)
    action_items: List[ActionItem]
    ethics_notes: str = Field(description="Fairness / privacy / due-process remarks, max 60 words")


class AgentStep(BaseModel):                         # what the agent says on every turn
    thought: str = Field(description="One short sentence of reasoning")
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    report: Optional[ComplianceReport] = None       # only when action == "final"


print("✓ Section C schemas defined")


# ----------------------------------------------------------------------------------------------------
# C · The tools (plain Python rule engine)   [notebook cell 23]
# ----------------------------------------------------------------------------------------------------
ETH_REQUIRED = ["conflicts", "transparency", "procurement", "data_protection"]
ETH_REFRESH_DAYS = 365
ETH_WARNING_DAYS = 30
ETH_TODAY = local_now().date()          # change to date(2026, 1, 1) etc. to test a specific reference date

ETH_ROSTER: List[Official] = []   # filled with load_eth_roster(); never rebound (tools read this list)


def _status_for(o: Official):
    missing = [c for c in ETH_REQUIRED if c not in o.completed]
    if o.last_training:
        days_since = (ETH_TODAY - date.fromisoformat(o.last_training)).days
        days_until_due = ETH_REFRESH_DAYS - days_since
    else:
        days_since, days_until_due = None, None
    overdue = days_until_due is not None and days_until_due < 0
    if missing or overdue or o.last_training is None:
        status = "NON_COMPLIANT"
    elif days_until_due <= ETH_WARNING_DAYS:
        status = "AT_RISK"
    else:
        status = "COMPLIANT"
    return {"official": o.name, "department": o.department, "status": status, "missing_courses": missing,
            "days_since_training": days_since, "days_until_refresher_due": days_until_due}


def eth_list_officials():
    return [o.name for o in ETH_ROSTER]


def eth_check_official(name):
    for o in ETH_ROSTER:
        if o.name.lower() == str(name).strip().lower():
            return _status_for(o)
    return {"error": f"unknown official '{name}'", "known": eth_list_officials()}


def eth_check_all():
    return [_status_for(o) for o in ETH_ROSTER]


def eth_team_summary():
    rows = eth_check_all()
    counts = {s: sum(r["status"] == s for r in rows) for s in ["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]}
    total = len(rows)
    # AT_RISK people are compliant (refresher due soon), so they count toward the compliance rate.
    rate = round(100 * (counts["COMPLIANT"] + counts["AT_RISK"]) / total, 1) if total else 0.0
    current = round(100 * counts["COMPLIANT"] / total, 1) if total else 0.0
    return {"total": total, **counts, "compliance_rate_pct": rate, "fully_current_pct": current}


ETH_TOOLS = {
    "list_officials": (eth_list_officials, "no args. Returns the names of all officials."),
    "check_official": (eth_check_official, 'args: {"name": "<official name>"}. Returns status, missing courses, days until refresher.'),
    "check_all":      (eth_check_all,      "no args. Returns status for every official (use this first for team questions)."),
    "team_summary":   (eth_team_summary,   "no args. Returns counts per status and the compliance rate."),
}


def load_eth_roster(rows):
    """Replace the roster contents in place."""
    new = [Official(**d) for d in rows]
    if not new:
        raise ValueError("The roster is empty: add at least one person.")
    ETH_ROSTER.clear()
    ETH_ROSTER.extend(new)
    return len(ETH_ROSTER)


print("✓ Section C tools ready:", ", ".join(ETH_TOOLS))


# ----------------------------------------------------------------------------------------------------
# C · The single agent (think → act → observe loop) and guardrail   [notebook cell 24]
# ----------------------------------------------------------------------------------------------------
ETH_TRACE = []
ETH_MAX_STEPS = 8


class ComplianceAgent:
    def __init__(self):
        tool_docs = "\n".join(f"- {name}: {doc}" for name, (_, doc) in ETH_TOOLS.items())
        self.system = (
            "You are the Ethics Training Compliance Agent for a public-transit agency. The people on the roster review and "
            "approve service-disruption actions, so their ethics training must be current.\n"
            "You do NOT calculate dates or statuses yourself: you call tools and report what they return.\n\n"
            f"TOOLS:\n{tool_docs}\n\n"
            "Each turn reply with ONE JSON object:\n"
            '  {"thought": "...", "action": "tool", "tool": "<tool name>", "args": {...}}   to call a tool, or\n'
            '  {"thought": "...", "action": "final", "report": {...ComplianceReport...}}    when you have what you need.\n'
            "Rules: call check_all (or team_summary) before answering team-level questions; include EVERY official in action_items; "
            "priority high = NON_COMPLIANT, medium = AT_RISK, low = COMPLIANT; keep tone neutral and fair (same standards for everyone); "
            "never speculate about reasons for non-compliance. Copy status and missing_courses EXACTLY from the tool output.\n"
            "Example of a final turn:\n"
            '{"thought": "done", "action": "final", "report": {"answer": "...", "compliance_rate_pct": 40.0, '
            '"action_items": [{"official": "A", "status": "NON_COMPLIANT", "missing_courses": ["procurement"], '
            '"action": "Complete procurement course", "priority": "high"}], "ethics_notes": "..."}}'
        )

    def run(self, question, verbose=True):
        if not ETH_ROSTER:                                   # never answer about an empty roster
            raise ValueError("The roster is empty — load it with load_eth_roster() first.")
        ETH_TRACE.clear()
        seen_calls = set()
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Question: {question}\nToday's date: {ETH_TODAY.isoformat()}"}]
        for step in range(1, ETH_MAX_STEPS + 1):
            t0 = time.time()
            raw = call_llm(messages, schema=AgentStep)
            try:
                s = AgentStep.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                if verbose:
                    print(f"Step {step}: ⚠ invalid output, asking the agent to fix it")
                ETH_TRACE.append({"step": step, "type": "invalid", "detail": str(e)[:80], "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid output: {str(e)[:400]}. Reply with corrected JSON only."}]
                continue
            if s.action == "tool":
                fn = ETH_TOOLS.get(s.tool or "")
                if fn is None:
                    result = {"error": f"unknown tool '{s.tool}'", "available": list(ETH_TOOLS)}
                elif (s.tool, json.dumps(s.args, sort_keys=True)) in seen_calls:
                    # a repeated call adds nothing: point the model back to the result it already has
                    result = {"note": f"You already called {s.tool} with these arguments; its result is above. Do not call "
                                      "tools again. Reply now with action \"final\" and the full report."}
                else:
                    seen_calls.add((s.tool, json.dumps(s.args, sort_keys=True)))
                    try:
                        result = fn[0](**s.args)
                    except TypeError as e:
                        result = {"error": f"bad arguments: {e}"}
                if verbose:
                    print(f"Step {step}: 🧠 {s.thought}\n         🔧 {s.tool}({json.dumps(s.args)}) ✓")
                ETH_TRACE.append({"step": step, "type": "tool", "detail": f"{s.tool}({json.dumps(s.args)})", "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "TOOL RESULT:\n" + json.dumps(result)}]
            elif s.report is not None:
                ETH_TRACE.append({"step": step, "type": "final", "detail": "ComplianceReport", "seconds": round(time.time() - t0, 1)})
                return s.report
            else:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": 'action "final" needs a "report" object. Try again.'}]
        raise RuntimeError(f"Agent did not finish within {ETH_MAX_STEPS} steps. Try a stronger model.")


def verify_report(report):
    """GUARDRAIL: compare the LLM's report with the deterministic rule engine."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    issues = []
    if not truth:
        issues.append("the roster is empty — nothing can be verified")
    seen = set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            issues.append(f"'{item.official}' is not on the roster (possible hallucination)")
            continue
        seen.add(item.official.lower())
        if item.status != t["status"]:
            issues.append(f"{item.official}: report says {item.status}, rules say {t['status']}")
        if set(item.missing_courses) != set(t["missing_courses"]):
            issues.append(f"{item.official}: missing courses differ (report {item.missing_courses} vs rules {t['missing_courses']})")
    for name in truth.keys() - seen:
        issues.append(f"{truth[name]['official']} is missing from the report")
    true_rate = eth_team_summary()["compliance_rate_pct"]
    if abs(report.compliance_rate_pct - true_rate) > 1:
        issues.append(f"compliance rate: report {report.compliance_rate_pct}% vs rules {true_rate}%")
    return issues


def reconcile_report(report):
    """GUARDRAIL: the rule engine owns the facts (status, missing courses, priority, rate). Overwrite them; keep the LLM's wording."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    prio = {"NON_COMPLIANT": "high", "AT_RISK": "medium", "COMPLIANT": "low"}
    fixed, seen = [], set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            continue                                   # drop officials the LLM made up
        seen.add(item.official.lower())
        fixed.append(item.model_copy(update={"status": t["status"], "missing_courses": t["missing_courses"],
                                             "priority": prio[t["status"]]}))
    for name, t in truth.items():                      # add officials the LLM forgot
        if name not in seen:
            fixed.append(ActionItem(official=t["official"], status=t["status"], missing_courses=t["missing_courses"],
                                    action="(added by guardrail) see rule-engine status", priority=prio[t["status"]]))
    s = eth_team_summary()
    answer, notes = report.answer, []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", report.answer):
        v = float(m.group(1))
        if abs(v - s["compliance_rate_pct"]) <= 1:
            continue
        if abs(v - s["fully_current_pct"]) <= 1:
            notes.append(f"{m.group(0)} is the fully-current share (COMPLIANT only), not the compliance rate")
        else:
            notes.append(f"{m.group(0)} is not supported by the rule engine")
    verified = (f"[Rule engine] Compliance rate {s['compliance_rate_pct']}% (COMPLIANT + AT_RISK, {s['COMPLIANT'] + s['AT_RISK']} of "
                f"{s['total']}); fully current {s['fully_current_pct']}%; NON_COMPLIANT {s['NON_COMPLIANT']}.")
    if notes:
        answer = verified + " [Guardrail correction: " + "; ".join(notes) + ".] LLM explanation: " + report.answer
    return report.model_copy(update={"action_items": fixed, "compliance_rate_pct": s["compliance_rate_pct"], "answer": answer})


eth_agent = ComplianceAgent()
ETH_WORKFLOW = [("in", "Roster +\nQuestion", "input"), ("agent", "Compliance\nAgent (think)", "llm"),
                ("tools", "Tools\n(act / observe)", "tool"), ("final", "Structured\nReport", "llm"),
                ("guard", "Guardrail\nverify", "tool"), ("out", "Verified\nOutput", "output")]
print("✓ Section C agent + guardrail ready")


# ----------------------------------------------------------------------------------------------------
# C · Rosters: transit review team (used by the governance gate) + the Section C sample rosters   [notebook cell 25]
# ----------------------------------------------------------------------------------------------------
def days_ago(n):
    return (ETH_TODAY - timedelta(days=n)).isoformat()


ALL4 = ["conflicts", "transparency", "procurement", "data_protection"]

ETH_SAMPLES = {
    # INTEGRATION: the people who act in this system (fictional). Statuses cover every case for testing the gate.
    "Transit review team (this system)": [
        {"name": "Ana Lopez",    "department": "Operations Supervisor (first reviewer)",  "completed": ALL4, "last_training": days_ago(90)},
        {"name": "Ben Carter",   "department": "Service Planner",                          "completed": ALL4, "last_training": days_ago(345)},
        {"name": "Chen Wei",     "department": "Operations Supervisor (second reviewer)", "completed": ALL4, "last_training": days_ago(30)},
        {"name": "Dana Brooks",  "department": "Policy Owner",                             "completed": ALL4, "last_training": days_ago(120)},
        {"name": "Eli Turner",   "department": "Operations Supervisor",                    "completed": ["conflicts", "transparency"], "last_training": days_ago(60)},
        {"name": "Farah Haddad", "department": "Operations Staff",                         "completed": ALL4, "last_training": days_ago(400)},
        {"name": "Gabe Ortiz",   "department": "Service Planner",                          "completed": ALL4, "last_training": days_ago(200)},
    ],
    # Section C sample rosters (fictional)
    "Mixed team (all statuses)": [
        {"name": "Alice Morgan",  "department": "Legal",       "completed": ALL4,                         "last_training": days_ago(90)},
        {"name": "Bob Rivera",    "department": "Procurement", "completed": ["conflicts", "transparency"], "last_training": days_ago(120)},
        {"name": "Carla Singh",   "department": "Finance",     "completed": ALL4,                         "last_training": days_ago(400)},
        {"name": "Dev Patel",     "department": "IT",          "completed": ALL4,                         "last_training": days_ago(345)},
        {"name": "Erin Walsh",    "department": "Operations",  "completed": [],                           "last_training": None},
    ],
    "Fully compliant office": [
        {"name": "Frank Osei",    "department": "Legal",   "completed": ALL4, "last_training": days_ago(30)},
        {"name": "Grace Kim",     "department": "Finance", "completed": ALL4, "last_training": days_ago(150)},
        {"name": "Hector Diaz",   "department": "IT",      "completed": ALL4, "last_training": days_ago(200)},
    ],
    "Overdue refreshers (new hires vs. veterans)": [
        {"name": "Irene Novak",   "department": "Procurement", "completed": ALL4,                     "last_training": days_ago(500)},
        {"name": "Jamal Brooks",  "department": "Procurement", "completed": ALL4,                     "last_training": days_ago(700)},
        {"name": "Keiko Tanaka",  "department": "Legal",       "completed": ["conflicts"],            "last_training": days_ago(20)},
        {"name": "Liam O'Brien",  "department": "Finance",     "completed": ["conflicts", "procurement", "transparency"], "last_training": days_ago(60)},
    ],
    "Approaching deadlines (early intervention)": [
        {"name": "Mona Haddad",   "department": "Operations", "completed": ALL4, "last_training": days_ago(350)},
        {"name": "Noah Fischer",  "department": "IT",         "completed": ALL4, "last_training": days_ago(340)},
        {"name": "Olivia Grant",  "department": "Legal",      "completed": ALL4, "last_training": days_ago(100)},
    ],
}
ETH_DEFAULT_QUESTION = "Give me the compliance status of the team and tell me who needs action first and why."

load_eth_roster(ETH_SAMPLES["Transit review team (this system)"])
print(f"✓ Roster loaded: {len(ETH_ROSTER)} people · summary {eth_team_summary()}")
display_df = pd.DataFrame(eth_check_all())
print(display_df[["official", "department", "status", "missing_courses", "days_until_refresher_due"]].to_string(index=False))
