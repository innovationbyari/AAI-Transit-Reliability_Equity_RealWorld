"""Section E4 · Outputs: interactive map, tables and charts, export

Generated from the notebook (cells 100). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# E4 · Outputs: interactive map, tables and charts   [notebook cell 100]
# ----------------------------------------------------------------------------------------------------
import folium
import plotly.graph_objects as go

STATUS_COLOR = {"MONITORING": "#6b7280", "CONFLICT_REVIEW": "#d97706", "SAFE_DEFAULT": "#2563eb", "AWAITING_REVIEW": "#dc2626",
                "AWAITING_SECOND_REVIEW": "#b91c1c", "AWAITING_IMPLEMENTATION": "#7c3aed", "IMPLEMENTED": "#059669",
                "EVALUATED": "#047857", "CLOSED_FALSE_ALARM": "#9ca3af"}
DEST_ICON = {"hospital": "plus-sign", "dialysis": "tint", "school": "book", "employment_center": "briefcase", "clinic": "heart", "retail": "shopping-cart"}


def _vcolor(v):
    # light → dark purple ramp for area vulnerability (area-level aggregate)
    ramp = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]
    return ramp[min(len(ramp) - 1, int(v * len(ramp)))]


def area_vulnerability(a, policy):
    b, w = policy["bounds"], policy["V_weights"]
    return (w["N"] * n_benefit(a["transit_dependence"], *b["transit_dependence"]) + w["I"] * n_benefit(a["income_disadvantage"], *b["income_disadvantage"])
            + w["B"] * n_benefit(a["disability"], *b["disability"])) / (w["N"] + w["I"] + w["B"])


def add_basemap(m):
    """CARTO Positron with the CARTO_API_KEY secret (free key: carto.com/basemaps/apikey); OpenStreetMap otherwise."""
    key = read_secret("CARTO_API_KEY")
    if key:
        folium.TileLayer(
            tiles="https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png?key=" + key,
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="CARTO Positron", max_zoom=20, subdomains="abcd").add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)


def build_map(case):
    pol = APP.PM.current
    m = folium.Map(location=list(CITY_CENTER), zoom_start=13, tiles=None, control_scale=True)
    add_basemap(m)
    fg_area = folium.FeatureGroup(name="Community areas (vulnerability, area-level)", show=True)
    for a in NET["areas"].values():
        v = area_vulnerability(a, pol)
        folium.Rectangle(bounds=a["bounds"], color="#ffffff", weight=1, fill=True, fill_color=_vcolor(v), fill_opacity=0.45,
                         tooltip=f"{a['name']}: transit dependence {a['transit_dependence']:.0%}, income disadvantage {a['income_disadvantage']:.0%}, "
                                 f"disability {a['disability']:.0%} · vulnerability index {v:.2f}").add_to(fg_area)
    fg_area.add_to(m)
    affected = set(case["raw"]["affected_stop_ids"]) if case else set()
    rid = case["raw"]["route_id"] if case else None
    fg_routes = folium.FeatureGroup(name="Routes and stops", show=True)
    for r in NET["routes"].values():
        hot = r["id"] == rid
        folium.PolyLine(r["path"], color=r["color"], weight=7 if hot else 4, opacity=0.95 if hot else 0.55,
                        dash_array="8 6" if r["mode"] == "rail" else None,
                        tooltip=f"{r['id']} {r['name']} ({r['mode']}, every {r['headway_min']} min)").add_to(fg_routes)
        for s in r["stops"]:
            on = s["stop_id"] in affected
            folium.CircleMarker((s["lat"], s["lon"]), radius=6 if on else 3, color="#b91c1c" if on else r["color"], fill=True,
                                fill_opacity=0.9, tooltip=f"{s['name']} · {s['stop_id']}" + (" · AFFECTED" if on else "")).add_to(fg_routes)
    fg_routes.add_to(m)
    fg_dest = folium.FeatureGroup(name="Essential destinations", show=True)
    for d in NET["destinations"].values():
        folium.Marker((d["lat"], d["lon"]), tooltip=f"{d['name']} ({d['type'].replace('_', ' ')})",
                      icon=folium.Icon(color="darkred" if d["type"] in ("hospital", "dialysis") else "cadetblue", icon=DEST_ICON[d["type"]])).add_to(fg_dest)
    for d in NET["depots"].values():
        folium.Marker((d["lat"], d["lon"]), tooltip=f"{d['name']}: {d['spare_buses']} spare bus(es), {d['operators']} operator(s)",
                      icon=folium.Icon(color="gray", icon="home")).add_to(fg_dest)
    fg_dest.add_to(m)
    if case:
        A = case["analysis"]
        stops = [NET["stops"][s] for s in case["raw"]["affected_stop_ids"]]
        mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
        fg_case = folium.FeatureGroup(name=f"{case['case_id']} incident and options", show=True)
        f = A.get("fields", {})
        sc = A.get("scores") or {}
        pop = (f"<b>{case['case_id']}</b><br>{_html.escape(case['scenario_name'])}<br>Status: <b>{case['status']}</b><br>"
               f"Delay: {f.get('delay_min', '—')} min · gap {f.get('headway_ratio', '—')}× · pass-ups {f.get('passups', '—')}<br>"
               f"C={A['confidence']['C'] if A.get('confidence') else '—'} · R={sc.get('R', '—')} V={sc.get('V', '—')} A={sc.get('A', '—')} P={sc.get('P', '—')}")
        folium.Marker(mid, popup=folium.Popup(pop, max_width=320), tooltip="Incident",
                      icon=folium.Icon(color="red" if case["status"] in REVIEW_STATUSES else "orange", icon="warning-sign")).add_to(fg_case)
        # Vehicle with delay
        v0 = stops[len(stops) // 2]
        folium.CircleMarker((v0["lat"] + 0.0012, v0["lon"]), radius=9, color="#111827", fill=True, fill_color="#facc15", fill_opacity=1,
                            tooltip=f"Vehicle on {case['raw']['route_id']}: {f.get('delay_min', '?')} min late, load {f.get('load_factor', '?')}").add_to(fg_case)
        chosen = case.get("approved_option_id")
        for o in A.get("options", []):
            if not o["feasible"]:
                continue
            style = {"weight": 5 if o["option_id"] == chosen else 3, "opacity": 0.95 if o["option_id"] == chosen else 0.6}
            if o["type"] == "dispatch_second_bus":
                dep = NET["depots"][o["option_id"].split("@")[1]]
                folium.PolyLine([(dep["lat"], dep["lon"]), mid], color="#16a34a", dash_array="6 6", tooltip=o["label"], **style).add_to(fg_case)
            elif o["type"] == "reroute":
                a, b = (stops[0]["lat"], stops[0]["lon"]), (stops[-1]["lat"], stops[-1]["lon"])
                off = 0.009
                folium.PolyLine([a, (a[0] + off, a[1] + off), (b[0] + off, b[1] + off), b], color="#ea580c", dash_array="4 6",
                                tooltip=o["label"], **style).add_to(fg_case)
            elif o["type"] == "protect_connection":
                ts = NET["stops"][case["raw"]["connections"]["transfer_stop_id"]]
                folium.CircleMarker((ts["lat"], ts["lon"]), radius=12, color="#0891b2", fill=False, weight=3, tooltip=o["label"]).add_to(fg_case)
        fg_case.add_to(m)
        m.fit_bounds([[min(s["lat"] for s in stops) - 0.02, min(s["lon"] for s in stops) - 0.03],
                      [max(s["lat"] for s in stops) + 0.02, max(s["lon"] for s in stops) + 0.03]])
    folium.LayerControl(collapsed=True).add_to(m)
    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:white;padding:8px 10px;border-radius:6px;"
              "font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'><b>Synthetic network (fictional)</b><br>"
              "<span style='color:#b91c1c'>●</span> affected stops &nbsp;<span style='color:#16a34a'>- -</span> dispatch path<br>"
              "<span style='color:#ea580c'>- -</span> reroute &nbsp;<span style='color:#0891b2'>○</span> protected transfer<br>"
              "Shading = area vulnerability (darker = higher)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


def map_iframe(case, height=540):
    doc = build_map(case).get_root().render()
    return f'<iframe srcdoc="{_html.escape(doc, quote=True)}" style="width:100%;height:{height}px;border:0;border-radius:8px"></iframe>'


def ph(df, msg="—"):
    return df if (df is not None and len(df)) else pd.DataFrame({"info": [msg]})


def evidence_df(case):
    if not case:
        return ph(None)
    rows = [{"source_id": e["source_id"], "source_type": e["source_type"], "status": e["status"], "quality": e.get("quality"),
             "timestamp_utc": e["timestamp"][:19], "age_s": e["age_s"],
             "fields": ", ".join(f"{k}={v}" for k, v in e["fields"].items()), "note": e["note"]} for e in case["analysis"]["evidence"]]
    return ph(pd.DataFrame(rows))


def exceptions_df(case):
    if not case:
        return ph(None)
    return ph(pd.DataFrame(case["analysis"]["exceptions"]), "No exception paths taken")


def trace_md(case):
    if not case:
        return "_No case yet._"
    icon = lambda r: "🟥" if any(k in r.upper() for k in ("SAFE", "NO", "MONITOR", "CONFLICT", "FAIL")) else "🟩"
    lines = [f"### Pipeline trace — {case['case_id']}", f"**Gate:** `{case['analysis']['gate']}` — {case['analysis']['gate_reason']}", ""]
    lines += [f"{icon(t['result'])} **{t['stage']}** · {t['result']}" + (f"  \n<small>{_html.escape(t['detail'])}</small>" if t["detail"] else "")
              for t in case["analysis"]["trace"]]
    if case.get("safe_default_reason"):
        lines.append(f"\n🔵 **Safe default:** {case['safe_default_reason']}")
    return "\n\n".join(lines)


def conf_df(case):
    if not case or not case["analysis"].get("confidence"):
        return ph(None, "Confidence not computed (agent safe mode)")
    c = case["analysis"]["confidence"]
    w = get_case_policy(case)["confidence_weights"]
    rows = [{"component": k, "meaning": m, "value": c[k], "weight": w[k], "contribution": r3(w[k] * c[k])}
            for k, m in [("Q", "source quality"), ("F", "freshness"), ("G", "cross-source agreement"), ("M", "completeness")]]
    rows.append({"component": "penalty", "meaning": "unresolved conflict", "value": c["penalty"], "weight": -1, "contribution": -c["penalty"]})
    rows.append({"component": "C", "meaning": f"threshold {get_case_policy(case)['confidence_threshold']}", "value": c["C"], "weight": None, "contribution": c["C"]})
    return pd.DataFrame(rows)


def norm_df(case):
    if not case or not case["analysis"].get("normalization"):
        return ph(None, "Impact scores not computed")
    return pd.DataFrame(case["analysis"]["normalization"])[["group", "component", "feature", "raw", "L", "U", "method", "normalized", "weight", "contribution", "note"]]


def rvap_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("scores"):
        s = case["analysis"]["scores"]
        pol = get_case_policy(case)
        fig.add_bar(x=["R reliability", "V vulnerability", "A access", "P priority"], y=[s["R"], s["V"], s["A"], s["P"]],
                    marker_color=["#2563eb", "#7c3aed", "#0891b2", "#111827"], text=[f"{v:.3f}" for v in (s["R"], s["V"], s["A"], s["P"])],
                    textposition="outside")
        fig.add_hline(y=pol["significance_R"], line_dash="dash", line_color="#dc2626", annotation_text=f"R significance {pol['significance_R']}")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_range=[0, 1.1], title="Component scores (0–1)", plot_bgcolor="white")
    return fig


def sig_md(case):
    if not case:
        return ""
    A = case["analysis"]
    trig = "\n".join(f"- **{t['trigger']}** (source: {t['source']}{', official ✔' if t['official'] else ''})" for t in A["hard_triggers"]) or "- none"
    miss = ", ".join(A["missing"]) or "none"
    return (f"### Significance\n**Result:** {'SIGNIFICANT' if A['significant'] else 'not significant'} · tier **{A['tier'] or '—'}**\n\n"
            f"**Hard triggers (bypass R threshold):**\n{trig}\n\n**Missing values (not imputed):** {miss}")


def options_df(case):
    if not case or not case["analysis"].get("ranking"):
        return ph(None, "No ranked options (case not significant, or no feasible option)")
    return pd.DataFrame([{k: r[k] for k in ["rank", "option_id", "label", "DR", "EB", "AP", "OF", "CE", "Ci", "Si", "cost_usd"]} for r in case["analysis"]["ranking"]])


def infeasible_df(case):
    if not case:
        return ph(None)
    inf = [{"option": o["label"], "reason": o["infeasible_reason"]} for o in case["analysis"].get("options", []) if not o["feasible"]]
    return ph(pd.DataFrame(inf), "All generated options are feasible")


def options_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("ranking"):
        rk = case["analysis"]["ranking"]
        labels = [f"#{r['rank']} {r['label'][:34]}" for r in rk][::-1]
        colors = {"DR": "#2563eb", "EB": "#7c3aed", "AP": "#0891b2", "OF": "#16a34a", "CE": "#f59e0b"}
        for c in ["DR", "EB", "AP", "OF", "CE"]:
            fig.add_bar(y=labels, x=[r["weighted"][c] for r in rk][::-1], name=c, orientation="h", marker_color=colors[c])
        fig.update_layout(barmode="stack")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=20), title="Si = Ci × weighted components (stacked)", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def decisions_df(case):
    if not case or not case["decisions"]:
        return ph(None, "No human decisions yet")
    return pd.DataFrame(case["decisions"])[["ts", "stage", "reviewer", "decision", "option_id", "recommended_option_id", "override",
                                             "response_s", "tier", "deadline_at", "revision_no", "reason"]]


def deadline_md(case):
    if not case:
        return "_No active case._"
    st = case["status"]
    if st in REVIEW_STATUSES:
        left = APP.seconds_left(case)
        mm, ss = divmod(max(0, int(left)), 60)
        r = case["review"]
        who = "second reviewer" if st == "AWAITING_SECOND_REVIEW" else "first reviewer"
        bar = "🟥" if left < 60 * DEMO_DEADLINE_SCALE else "🟧" if left < 180 * DEMO_DEADLINE_SCALE else "🟩"
        extra = " · ⚠️ **No feasible option** — revise or reject" if r.get("no_feasible_option") else ""
        return (f"### {bar} {mm:02d}:{ss:02d} left — waiting for {who}\nTier **{r['tier']}** ({r['deadline_min']} min, covers both reviewers) · "
                f"revisions used {case['revisions']}/{get_case_policy(case)['max_revisions']}{extra}. No response → safe default.")
    return f"**Status:** `{st}` — {STATUS_HELP.get(st, '')}"


def outcome_df(case):
    if not case or not case.get("outcome"):
        return ph(None, "No outcome evaluated yet")
    o = case["outcome"]
    if "pred_delay_reduction_min" not in o:
        return pd.DataFrame([o])
    rows = [("Delay reduction (min)", o["pred_delay_reduction_min"], o["obs_delay_reduction_min"]),
            ("Access protection (0–1)", o["pred_access"], o["obs_access"]),
            ("Equity: vulnerable riders served", o["pred_equity_riders"], o["obs_equity_riders"]),
            ("Cost (USD)", o["pred_cost_usd"], o["actual_cost_usd"])]
    df = pd.DataFrame([{"metric": m, "predicted": p, "observed": ob, "difference": r3((ob or 0) - (p or 0))} for m, p, ob in rows])
    extra = pd.DataFrame([{"metric": "Observed data source", "predicted": "—", "observed": o.get("observed_source", "entered by a person"), "difference": "—"},
                          {"metric": "False alarm", "predicted": "—", "observed": o["false_alarm"], "difference": "—"},
                          {"metric": "Rider-reported impact (1 low – 5 high)", "predicted": "—", "observed": o["rider_reported_impact"], "difference": "—"}])
    return pd.concat([df.astype(object), extra], ignore_index=True)


def outcome_plot(case):
    fig = go.Figure()
    o = case.get("outcome") if case else None
    if o and "pred_delay_reduction_min" in o:
        m = ["Delay reduction", "Access", "Equity riders", "Cost"]
        pr = [o["pred_delay_reduction_min"], o["pred_access"], o["pred_equity_riders"], o["pred_cost_usd"]]
        ob = [o["obs_delay_reduction_min"], o["obs_access"], o["obs_equity_riders"], o["actual_cost_usd"]]
        fig.add_bar(x=m, y=[1.0 if p else 0 for p in pr], name="predicted (=1.0)", marker_color="#94a3b8")
        fig.add_bar(x=m, y=[(b / p) if p else 0 for p, b in zip(pr, ob)], name="observed / predicted", marker_color="#2563eb")
    fig.update_layout(height=300, barmode="group", margin=dict(l=20, r=20, t=30, b=20), title="Observed relative to predicted", plot_bgcolor="white")
    return fig


def versions_df():
    return pd.DataFrame([{"version": v["version"], "released_at": v["released_at"], "approved_by": v["approved_by"], "reason": v["reason"],
                          "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in v["changes"]) or "—",
                          "backtest": v["backtest_summary"] or "—", "active": i == APP.PM.current_index}
                         for i, v in enumerate(APP.PM.versions)])


def proposals_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in p["changes"]),
                             "evidence": p["evidence"], "rationale": p["rationale"], "status": p["status"]} for p in APP.proposals]),
              "No open proposals — run the Insight Agent")


def proposal_log_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "status": p["status"], "decided_by": p.get("decided_by"), "reason": p.get("reason"),
                             "backtest": p.get("backtest", "—")} for p in APP.proposal_log]), "No decisions yet")


def audit_df(case_only=False):
    rows = APP.audit if not case_only or not APP.active else APP.active["audit"]
    return ph(pd.DataFrame(rows[::-1][:500]), "Audit log empty")


def cases_df():
    rows = [{"case_id": c["case_id"], "created_at": c["created_at"][:19], "scenario": c["scenario_name"], "status": c["status"],
             "gate": c["analysis"]["gate"], "P": (c["analysis"].get("scores") or {}).get("P"), "tier": c["analysis"].get("tier"),
             "rules": c["policy_version"], "approved_option": c.get("approved_option_id")} for c in APP.cases.values()]
    return ph(pd.DataFrame(rows[::-1]), "No live cases yet")


def monitor_df():
    rows = [{"case_id": c["case_id"], "status": c["status"], "route": c["raw"]["route_id"],
             "confidence": (c["analysis"].get("confidence") or {}).get("C"), "priority_P": (c["analysis"].get("scores") or {}).get("P"),
             "reason": c.get("safe_default_reason") or c["analysis"]["gate_reason"]}
            for c in APP.cases.values() if c["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    return ph(pd.DataFrame(rows), "Monitoring queue empty")


def alerts_df(case):
    return ph(pd.DataFrame(case["alerts"]) if case else None, "No alerts")


def export_outputs():
    """Write every Colab output (shared case state, evidence, exceptions, scores, scenarios, decisions, outcomes, audit, rules, map)."""
    ts = local_now().strftime("%Y%m%dT%H%M%S%Z")
    path = os.path.join(OUTPUT_DIR, f"transit_dashboard_outputs_{ts}.zip")
    cases = list(APP.cases.values())
    ev, ex, sc, op, dc, oc = [], [], [], [], [], []
    for c in cases:
        A = c["analysis"]
        ev += [{"case_id": c["case_id"], **{k: v for k, v in e.items() if k != "fields"}, "fields": json.dumps(e["fields"])} for e in A["evidence"]]
        ex += [{"case_id": c["case_id"], **e} for e in A["exceptions"]]
        if A.get("scores"):
            sc.append({"case_id": c["case_id"], **{k: A["scores"][k] for k in "RVAP"}, **A["scores"]["components"], **(A["confidence"] or {}), "tier": A["tier"]})
        op += [{"case_id": c["case_id"], **{k: v for k, v in r.items() if k not in ("weights", "weighted")}} for r in A.get("ranking", [])]
        dc += [{"case_id": c["case_id"], **d} for d in c["decisions"]]
        if c.get("outcome"):
            oc.append({"case_id": c["case_id"], **c["outcome"]})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("shared_case_state.json", json.dumps(cases, indent=1, default=str))
        z.writestr("policy_versions.json", json.dumps(APP.PM.versions, indent=1, default=str))
        z.writestr("insight_proposals.json", json.dumps(APP.proposals + APP.proposal_log, indent=1, default=str))
        for name, rows in [("verified_evidence", ev), ("exception_log", ex), ("component_scores", sc), ("scenario_table_and_rank", op),
                           ("human_decisions_and_timing", dc), ("outcome_evaluation", oc), ("audit_record", APP.audit)]:
            z.writestr(f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))
        z.writestr("reviewer_audit_rubber_stamp.csv", APP.rubber_stamp_report().to_csv(index=False))
        z.writestr("incident_history.csv", incident_history_df(APP).to_csv(index=False))
        z.writestr("section_c_ethics_training_status.csv", pd.DataFrame(eth_check_all()).to_csv(index=False))
        z.writestr("section_d_disclosures.csv", disclosures_df().to_csv(index=False))
        try:
            z.writestr("section_b_notebook_form_audit.csv", audit_table().to_csv(index=False))
        except NameError:
            pass
        if APP.active:
            z.writestr(f"map_{APP.active['case_id']}.html", build_map(APP.active).get_root().render())
    APP.log(None, "system", "system", "Export", "Outputs exported", path)
    return path
