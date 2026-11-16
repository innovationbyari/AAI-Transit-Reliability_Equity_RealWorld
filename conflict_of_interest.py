"""Section D · Conflict-of-interest detection (schemas, agents, disclosures)

Generated from the notebook (cells 28, 29, 30). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# D · Structured outputs (the "contract" between agents)   [notebook cell 28]
# ----------------------------------------------------------------------------------------------------
HOLDING_KINDS = ["direct_stock", "business_ownership", "family_business", "donation_received",
                 "blind_trust", "index_fund", "other"]


class Holding(BaseModel):
    entity: str = Field(description="Company / organisation name")
    kind: Literal["direct_stock", "business_ownership", "family_business", "donation_received",
                  "blind_trust", "index_fund", "other"]
    owner: Literal["official", "spouse", "family", "donor"] = "official"
    value_usd: Optional[float] = None
    note: str = ""


class Decision(BaseModel):
    bill: str = Field(description="Name of the bill / vote / decision")
    subject: str = Field(description="What it is about, in a few words")
    action: str = Field(default="", description="e.g. voted yes, signed contract, chaired hearing")


class ExtractedFacts(BaseModel):                       # Agent 1 output
    official_name: str
    role: str = ""
    holdings: List[Holding] = []
    decisions: List[Decision] = []
    relationships: List[str] = []


class Flag(BaseModel):
    type: Literal["financial_interest", "family_interest", "donor_influence", "other"]
    severity: Literal["low", "medium", "high"]
    entity: str
    decision: str
    reasoning: str
    evidence: List[str] = Field(default=[], description="Short quotes / facts from the case file")


class AnalystOutput(BaseModel):                        # Agent 3 output
    flags: List[Flag] = []
    overall_risk_score: float = Field(ge=0, le=1)
    dismissed_rule_hits: List[str] = Field(default=[], description="Rule-screener hits judged not to be conflicts, with reason")


class ReviewerOutput(BaseModel):                       # Agent 4 output
    verdict_on_analysis: Literal["agree", "partially_agree", "disagree"]
    innocent_explanations: List[str] = Field(default=[], description="Hypothetical explanations, each starting with 'Hypothetical:'")
    mitigating_factors: List[str] = Field(default=[], description="ONLY facts stated in the case file (e.g. recusal, blind trust, competitive bidding). Empty list if none. Never speculate.")
    questions_for_official: List[str] = []


class FinalReport(BaseModel):                          # Agent 5 output
    official: str
    recommendation: Literal["CLEAR", "MONITOR", "REVIEW", "ESCALATE"]
    risk_score: float = Field(ge=0, le=1)
    summary: str = Field(description="2-3 sentences (40-80 words): who, which holding, which decision, why it matters")
    key_flags: List[str] = []
    required_actions: List[str] = []
    due_process_note: str


print("✓ Section D schemas defined:", ", ".join(c.__name__ for c in [ExtractedFacts, AnalystOutput, ReviewerOutput, FinalReport]))


# ----------------------------------------------------------------------------------------------------
# D · The agents, the rule-screener tool, and the orchestrator   [notebook cell 29]
# ----------------------------------------------------------------------------------------------------
COI_TRACE = []
COI_TOTAL_STEPS = 5


def coi_traced(name, kind, fn, verbose=True):
    """Run one workflow step, print progress, record timing."""
    i = len(COI_TRACE) + 1
    if verbose:
        print(f"▶ Step {i}/{COI_TOTAL_STEPS}  {name:<22} [{kind}] ...", end=" ", flush=True)
    t0 = time.time()
    try:
        out = fn()
    except Exception:
        COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "failed"})
        if verbose:
            print("✗ failed")
        raise
    COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "ok"})
    if verbose:
        print(f"✓ {COI_TRACE[-1]['seconds']}s")
    return out


class COIAgent:                                         # renamed from `Agent` (clashed with Section B)
    def __init__(self, name, system, schema):
        self.name, self.system, self.schema = name, system, schema

    def run(self, user_message, verbose=True):
        def _call():
            obj, retries = run_structured(self.schema, self.system, user_message)
            return obj
        return coi_traced(self.name, "LLM agent", _call, verbose)


coi_extractor = COIAgent("Extractor Agent",
    "You are the Extractor Agent in a government-ethics review pipeline. Read the case file and extract ONLY "
    "facts that are explicitly stated: the official, their holdings/relationships (own, spouse, family, donors), "
    "and the votes/decisions they made or will make. Do not infer or invent. Use kind 'blind_trust' for blind trusts, "
    "'index_fund' for broad diversified funds, 'family_business' for a relative's company, 'donation_received' for donors.",
    ExtractedFacts)

coi_analyst = COIAgent("Conflict Analyst Agent",
    "You are the Conflict Analyst Agent. You receive extracted facts and the output of a deterministic rule screener. "
    "Decide which potential conflicts are REAL. For each flag give type, severity, the entity, the decision, your reasoning, "
    "and evidence taken from the facts. Severity guide: high = direct or family financial stake in the decision; "
    "medium = donor influence or sector-level exposure; low = remote exposure (blind trust, diversified fund). "
    "Put rule hits you judge to be false alarms in dismissed_rule_hits with a reason. If there is no conflict, return an empty flags list. "
    "overall_risk_score is 0 (none) to 1 (severe).",
    AnalystOutput)

coi_reviewer = COIAgent("Reviewer Agent",
    "You are the Reviewer Agent, a devil's advocate protecting due process. Challenge the analyst's flags. Keep FACTS and HYPOTHESES separate: "
    "mitigating_factors may contain ONLY things stated in the extracted facts (e.g. recusal, blind trust, competitive bidding, diversified fund); "
    "if the case states none, return an empty list. Never invent facts. Speculative alternatives go in innocent_explanations and must start with "
    "'Hypothetical:'. Write concrete questions the official should be allowed to answer. Say whether you agree, partially agree, or disagree "
    "with the analysis. Be fair, not lenient.",
    ReviewerOutput)

coi_decider = COIAgent("Decision Agent",
    "You are the Decision Agent. Combine the facts, rule screen, analysis and review into a final report. Recommendation guide: "
    "CLEAR = no material flags; MONITOR = only low-severity flags; REVIEW = medium/high flags with a stated mitigation; "
    "ESCALATE = high-severity flag with no stated mitigation. Only the reviewer's mitigating_factors count as mitigation; hypothetical explanations "
    "and unanswered questions do NOT lower the recommendation. The summary must be 2-3 sentences (40-80 words) naming the official, the holding, "
    "the decision and why it matters. required_actions must be concrete (e.g. 'request recusal statement'). due_process_note must state that the "
    "official may respond before any action and that a human decides.",
    FinalReport)


# ---------- the deterministic tool ----------
COI_SECTORS = {
    "technology": ["tech", "software", "semiconductor", "cloud", "cyber", "antitrust", "artificial intelligence", "data privacy"],
    "healthcare_pharma": ["pharma", "drug", "health", "medical", "biotech", "prescription", "medicare"],
    "energy": ["oil", "gas", "energy", "solar", "pipeline", "coal", "utility", "drilling"],
    "defense": ["defense", "weapon", "aerospace", "military"],           # "procurement" removed (sector-neutral)
    "finance": ["bank", "financ", "insur", "lending", "crypto"],
    # transportation / construction was missing, so transit vendors and road contracts were never screened
    "transportation_infrastructure": ["transit", "bus", "coach", "rail", "paratransit", "shuttle", "fleet", "depot",
                                      "highway", "road", "paving", "construction", "detour", "transportation"],
}
COI_WEIGHTS = {"direct_stock": 1.0, "business_ownership": 1.0, "family_business": 0.8, "donation_received": 0.7,
               "blind_trust": 0.4, "index_fund": 0.2, "other": 0.5}
COI_THRESHOLD = 0.7

_LEGAL_SUFFIX = r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|group|holdings)\b\.?"
_GENERIC_WORDS = {"construction", "services", "service", "sons", "and", "the", "of", "international", "systems",
                  "solutions", "partners", "company", "group", "transport", "transit", "coach", "&"}


def norm_entity(text):
    """normalize names so 'TechCorp Inc.' matches "TechCorp's cloud division"."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(_LEGAL_SUFFIX, " ", t)
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def name_matches(entity, text):
    e, t = norm_entity(entity), " " + norm_entity(text) + " "
    if e and f" {e} " in t:
        return True
    core = " ".join(w for w in e.split() if w not in _GENERIC_WORDS)    # e.g. "nair" from "Nair & Sons Construction LLC"
    return len(core) >= 3 and f" {core} " in t


def sectors_of(text):
    t = text.lower()
    return {s for s, kws in COI_SECTORS.items() if any(re.search(r"\b" + re.escape(k), t) for k in kws)}


def rule_screen(facts, decisions=None, threshold=None):
    """Deterministic screen: match each holding against each decision by name or by sector.
    INTEGRATION: `decisions` can be supplied (the options a reviewer is approving) instead of facts.decisions."""
    threshold = COI_THRESHOLD if threshold is None else threshold
    hits = []
    for h in facts.holdings:
        for d in (decisions if decisions is not None else facts.decisions):
            decision_text = f"{d.bill} {d.subject} {d.action}"
            name_match = name_matches(h.entity, decision_text)
            shared = sectors_of(f"{h.entity} {h.note}") & sectors_of(decision_text)
            if name_match or shared:
                score = round(COI_WEIGHTS[h.kind] * (1.0 if name_match else 0.75), 2)
                hits.append({"holding": h.entity, "kind": h.kind, "owner": h.owner, "decision": d.bill,
                             "match": "name" if name_match else f"sector:{'/'.join(sorted(shared))}", "score": score})
    top = max([x["score"] for x in hits], default=0.0)
    return {"hits": hits, "rule_score": top, "threshold": threshold, "over_threshold": top >= threshold}


# ---------- the orchestrator ----------
def coi_run_pipeline(case_text, verbose=True):
    COI_TRACE.clear()
    facts = coi_extractor.run(f"CASE FILE:\n{case_text}", verbose)
    screen = coi_traced("Rule Screener", "python tool", lambda: rule_screen(facts), verbose)
    analysis = coi_analyst.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                               "\n\nRULE SCREENER OUTPUT:\n" + json.dumps(screen, indent=1), verbose)
    review = coi_reviewer.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                              "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=1), verbose)
    final = coi_decider.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                            "\n\nRULE SCREEN:\n" + json.dumps(screen) +
                            "\n\nANALYST:\n" + analysis.model_dump_json() +
                            "\n\nREVIEWER:\n" + review.model_dump_json(), verbose)
    return {"facts": facts, "screen": screen, "analysis": analysis, "review": review, "final": final}


def coi_guardrail(result):
    """Consistency check on the final recommendation (two consistency checks + recommendation check)."""
    final, analysis, review, screen = result["final"], result["analysis"], result["review"], result["screen"]
    issues = []
    if any(f.severity == "high" for f in analysis.flags) and not review.mitigating_factors and final.recommendation != "ESCALATE":
        issues.append("high-severity flag with no stated mitigation, but the recommendation is not ESCALATE")
    if final.recommendation == "CLEAR" and analysis.flags:
        issues.append("recommendation is CLEAR although the analyst raised flags")
    if final.recommendation == "CLEAR" and screen["over_threshold"]:
        issues.append(f"recommendation is CLEAR although the deterministic rule score {screen['rule_score']} ≥ threshold {screen['threshold']}")
    return issues


def coi_enforce(result):
    """GUARDRAIL: when the consistency check fails, the stricter recommendation wins (a person still decides)."""
    issues, final = coi_guardrail(result), result["final"]
    rule = float(result["screen"].get("rule_score") or 0)
    if final.risk_score < rule:                     # the stated risk is never below the deterministic rule score
        final = final.model_copy(update={"risk_score": rule})
        issues = issues + [f"risk raised to the rule screener's {rule}"]
    if not any("ESCALATE" in i or "CLEAR" in i for i in issues):
        return final, issues
    new = "ESCALATE" if any("ESCALATE" in i for i in issues) else "REVIEW"
    return final.model_copy(update={"recommendation": new,
                                    "summary": final.summary + f" [Guardrail: changed {final.recommendation} → {new}.]"}), issues


COI_WORKFLOW = [("case", "Case file", "input"), ("Extractor Agent", "1 Extractor", "llm"),
                ("Rule Screener", "2 Rule\nScreener", "tool"), ("Conflict Analyst Agent", "3 Conflict\nAnalyst", "llm"),
                ("Reviewer Agent", "4 Reviewer", "llm"), ("Decision Agent", "5 Decision", "llm"),
                ("report", "Report", "output")]
print("✓ Section D: 4 agents + 1 tool + orchestrator ready")


# ----------------------------------------------------------------------------------------------------
# D · Disclosures: transit staff (used by the gate) + the Section D sample cases   [notebook cell 30]
# ----------------------------------------------------------------------------------------------------
COI_SAMPLES = {
    "Transit · planner reviews a dispatch to his spouse's depot contractor": """
Service planner Gabe Ortiz is the first reviewer on a bus disruption case. The top-ranked option dispatches a spare bus
from Depot B. Depot B is run under contract by Metro Coach Services LLC, which is paid per spare-bus dispatch. His spouse
is the majority owner of Metro Coach Services LLC. He is about to approve the Depot B dispatch and has not recused.
""",
    "Direct stock + related vote (clear conflict)": """
Senator Maria Alvarez sits on the Technology Oversight Committee. Her latest financial disclosure lists 12,000 shares of
TechCorp Inc. (value about $480,000) held directly in her own brokerage account. Last week she voted YES on the
Tech Antitrust Reform Act, which would exempt TechCorp's cloud division from a pending federal investigation.
She did not recuse herself and made no public statement about her holdings.
""",
    "Blind trust + broad regulation (ambiguous)": """
Senator David Chen owns a diversified portfolio inside a qualified blind trust managed by an independent trustee.
He knows the trust was originally funded with a mix of technology and healthcare stocks but cannot see current holdings.
He voted on the Digital Platform Regulation Act, which imposes new data privacy rules on large technology companies.
He has publicly disclosed the trust and stated that he does not direct any trades.
""",
    "Spouse's firm wins a contract (family interest)": """
Commissioner Priya Nair oversees the state Department of Transportation budget. Her spouse is the majority owner of
Nair & Sons Construction LLC. In March the department awarded Nair & Sons a $2.3 million road-repair contract; Priya
participated in the budget meeting that approved the appropriation for the Highway Repair Program and did not recuse.
The contract was competitively bid with three other bidders.
""",
    "Major donor, pharma vote (donor influence)": """
Representative James O'Connor received $250,000 in campaign contributions last cycle from executives and the PAC of
HealthFirst Pharmaceuticals, his single largest donor. This month he cast the deciding committee vote against the
Prescription Drug Price Cap Act, which HealthFirst lobbied heavily against. He owns only a small index fund
and has no direct holdings in any pharmaceutical company.
""",
    "Low-risk baseline (nothing suspicious)": """
Councilmember Lena Park holds a broad-market index fund in her retirement account (about $40,000) and owns no individual
stocks. She voted on a city budget amendment to repaint public libraries. No relatives work for the vendor. She received
no campaign donations from the vendor.
""",
}

# INTEGRATION: annual disclosures of the people who decide in this system (fictional). The structured facts are what
# the Extractor Agent would produce; they are stored so the gate works without a language model.
STAFF_DISCLOSURES = {
    "Gabe Ortiz": {
        "text": "Service planner Gabe Ortiz reviews disruption responses. His spouse is the majority owner of Metro Coach "
                "Services LLC, the contract operator of Depot B, which is paid per spare-bus dispatch.",
        "facts": ExtractedFacts(official_name="Gabe Ortiz", role="Service Planner", holdings=[
            Holding(entity="Metro Coach Services LLC", kind="family_business", owner="spouse",
                    note="contract operator of Depot B; paid per spare bus dispatch")],
            relationships=["spouse owns Metro Coach Services LLC"]),
    },
    "Chen Wei": {
        "text": "Operations supervisor Chen Wei holds a broad-market index fund (about $25,000) in a retirement account "
                "and no individual stocks.",
        "facts": ExtractedFacts(official_name="Chen Wei", role="Operations Supervisor",
                                holdings=[Holding(entity="Total Market Index Fund", kind="index_fund", value_usd=25000)]),
    },
    "Ana Lopez":   {"text": "Ana Lopez reports no holdings, family businesses or gifts.",   "facts": ExtractedFacts(official_name="Ana Lopez")},
    "Dana Brooks": {"text": "Dana Brooks reports no holdings, family businesses or gifts.", "facts": ExtractedFacts(official_name="Dana Brooks")},
    "Ben Carter":  {"text": "Ben Carter reports no holdings, family businesses or gifts.",  "facts": ExtractedFacts(official_name="Ben Carter")},
}

# Deterministic check of two Section D samples — no LLM needed
_alvarez = ExtractedFacts(official_name="Maria Alvarez", holdings=[Holding(entity="TechCorp Inc.", kind="direct_stock", value_usd=480000)],
                          decisions=[Decision(bill="Tech Antitrust Reform Act", subject="exempting TechCorp's cloud division from a pending federal investigation", action="voted yes")])
_nair = ExtractedFacts(official_name="Priya Nair", holdings=[Holding(entity="Nair & Sons Construction LLC", kind="family_business", owner="spouse")],
                       decisions=[Decision(bill="Highway Repair Program appropriation", subject="road-repair contract awarded to Nair & Sons", action="approved budget")])
_depot_b = [Decision(bill="Dispatch a second bus — Depot B", action="approve operational action",
                     subject="dispatch second bus for transit bus service; parties: Metro Coach Services LLC")]
for _who in ["Gabe Ortiz", "Chen Wei", "Ana Lopez"]:
    _s = rule_screen(STAFF_DISCLOSURES[_who]["facts"], _depot_b)
    print(f"{_who:10s} approving a Depot B dispatch → rule score {_s['rule_score']} "
          f"({'RECUSE' if _s['over_threshold'] else 'clear'}) {[h['holding'] for h in _s['hits']] or ''}")
print("Section D samples (stock vote, spouse contract) flagged by the screener:", rule_screen(_alvarez)["over_threshold"], rule_screen(_nair)["over_threshold"])
