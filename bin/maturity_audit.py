#!/usr/bin/env python3
"""maturity_audit.py — audit the Flesh Pits causal-maturity ledgers.

Reads prototypes/architecture-a/MATURITY.md and
prototypes/architecture-b/MATURITY.md, and for every component row:
  1. extracts the claimed level + every cited evidence token
     (experiment IDs, receipt filenames, NR refs, rerun claims);
  2. verifies each token against disk (receipt exists + non-empty,
     hash-chained where claimed; experiment script/function exists;
     NR full entry exists in a research/negative_results.md);
  3. checks level-justification consistency:
       REPRODUCED -> multi-seed repro receipt cited;
       CAUSAL     -> lesion/ablation/kill experiment cited;
       GENERALIZING -> generalization experiments cited;
       ADAPTIVE   -> experience-driven change evidence cited;
  4. checks ledger-internal consistency:
       component-table levels vs the "Maturity changes from this
       reproduction" section; kill-table verdicts vs replication
       outcomes; open-gaps wording vs repro receipts on disk.

Emits a machine-readable JSON report (default: maturity_audit_report.json
next to this script) and a human summary on stdout.
Exit 0 iff zero STALE and zero OVERCLAIMED rows.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FP = Path(__file__).resolve().parent.parent  # flesh-pits/

LEVELS = ["PRESENT", "EXECUTED", "INTEGRATED", "CAUSAL",
          "ADAPTIVE", "GENERALIZING", "REPRODUCED"]
RANK = {l: i for i, l in enumerate(LEVELS)}

TABLE_HEADER = re.compile(r'^\|\s*Component\s*\|\s*(Maturity|Level)\s*\|\s*Evidence\s*\|')
SEP_ROW = re.compile(r'^\|\s*-+')
NR_RE = re.compile(r'NR-[AB]-\d{3}')
RECEIPT_RE = re.compile(
    r'[A-Za-z0-9_\-\./]+\.(?:jsonl|ndjson|json)(?![A-Za-z])')
RECEIPT_PATH_RE = re.compile(
    r'flesh-pits/[A-Za-z0-9_\-\./]+\.(?:jsonl|ndjson|json)(?![A-Za-z])')
EXP_FP_RE = re.compile(r'EXP-[A-Z]+-[A-Z0-9]+')
# heading-level NR entry: the ID must head a section, not sit mid-sentence
NR_HEADING = re.compile(r'(?m)^\s*(?:#{1,4}\s+.*|[-*]\s+)?NR-[AB]-\d{3}\b')
NR_HEADING_STRICT = re.compile(r'(?m)^#{1,4}\s.*(NR-[AB]-\d{3})')


def parse_component_table(path):
    rows = []
    in_table = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if TABLE_HEADER.match(line):
            in_table = True
            continue
        if in_table:
            if SEP_ROW.match(line):
                continue
            if not line.startswith("|"):
                break
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) < 3:
                continue
            rows.append({"component": cols[0], "level_raw": cols[1],
                         "evidence": cols[2]})
    return rows


def base_level(level_raw):
    m = re.search(r'\b(' + "|".join(LEVELS) + r')\b', level_raw)
    return m.group(1) if m else None


ARCHES = {
    "a": {
        "maturity": FP / "prototypes/architecture-a/MATURITY.md",
        "receipts": FP / "prototypes/architecture-a/receipts",
        "exp_dir": FP / "prototypes/architecture-a/experiments",
        "nr_files": [FP / "prototypes/architecture-a/research/negative_results.md",
                     FP / "research/negative_results.md"],
        "exp_ids": ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"],
        "script_for": {
            "K1": "k1_capacity_lesion.py", "K2": "k2_broadcast_lesion.py",
            "K3": "k3_ignition_probe.py", "K4": "k4_attention_baseline.py",
            "K5": "k5_multitask_generalization.py",
            "K6": "k6_stationarity_resolution.py",
            "K7": "k7_bid_ablation.py", "K8": "k8_r1_wiring_confirmation.py",
            "IDENTITY": "identity_symmetry_check.py",
            "REPRO": "repro_multiseed_a.py",
        },
        "receipt_for": {
            "K1": "k1_capacity_lesion.json", "K2": "k2_broadcast_lesion.json",
            "K3": "k3_ignition_probe.json", "K4": "k4_attention_baseline.json",
            "K5": "k5_multitask_generalization.json",
            "K6": "k6_stationarity_resolution.json",
            "K7": "k7_bid_ablation.json",
            "K8": "k8_r1_wiring_confirmation.json",
        },
        "lesion_exps": {"K1", "K2", "K7"},
    },
    "b": {
        "maturity": FP / "prototypes/architecture-b/MATURITY.md",
        "receipts": FP / "prototypes/architecture-b/receipts",
        "exp_files": [FP / "prototypes/architecture-b/experiments.py",
                      FP / "prototypes/architecture-b/experiments_phase4.py",
                      FP / "prototypes/architecture-b/repro_multiseed.py"],
        "nr_files": [FP / "prototypes/architecture-b/research/negative_results.md",
                     FP / "research/negative_results.md"],
        "exp_ids": ["K1", "K2", "K3", "K4", "K5", "K3B", "M1",
                    "C1", "C2", "C2B", "C4", "C4B", "BAR"],
        # exp id -> (source file index, function name)
        "func_for": {
            "K1": (0, "exp_k1"), "K2": (0, "exp_k2"), "K3": (0, "exp_k3"),
            "K4": (0, "exp_k4"), "K5": (0, "exp_k5"), "C1": (0, "exp_c1"),
            "C2": (0, "exp_c2"), "C4": (0, "exp_c4"), "BAR": (0, "exp_bar"),
            "M1": (1, "exp_m1"), "C2B": (1, "exp_c2b"),
            "C4B": (1, "exp_c4b"), "K3B": (1, "exp_k3b"),
        },
        "lesion_exps": {"K1", "K2", "K3", "C2", "C4", "M1"},
    },
}

findings = []  # {arch, row, type, message}


def flag(arch, row, ftype, message):
    findings.append({"arch": arch, "row": row, "type": ftype,
                     "message": message})


def extract_tokens(evidence):
    toks = {"receipts": [], "nrs": [], "exps": [],
            "byte_identical": False, "bit_identical": False,
            "hash_chained": False}
    toks["receipts"] = RECEIPT_RE.findall(evidence)
    toks["receipt_paths"] = RECEIPT_PATH_RE.findall(evidence)
    toks["nrs"] = NR_RE.findall(evidence)
    toks["exps"] = EXP_FP_RE.findall(evidence)
    low = evidence.lower()
    toks["byte_identical"] = "byte-identical" in low
    toks["bit_identical"] = "bit-identical" in low
    toks["hash_chained"] = "hash-chained" in low
    return toks


def resolve_receipt(arch, name):
    cfg = ARCHES[arch]
    if name.startswith("flesh-pits/"):
        # workspace-relative path, e.g. flesh-pits/receipts/X.ndjson
        p = FP.parent / name
        if p.is_file():
            return p
    if "/" in name:
        for base in (FP, cfg["maturity"].parent):
            p = base / name
            if p.is_file():
                return p
        return None
    p = cfg["receipts"] / name
    return p if p.is_file() else None


def receipt_ok(path):
    if path is None or not path.is_file():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"
    try:
        if path.suffix in (".ndjson", ".jsonl"):
            n = sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip())
            if n == 0:
                return False, "empty"
        else:
            json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"unparseable: {e}"
    return True, "ok"


def check_hash_chain(path):
    """Verify prev_hash linkage across records (json: receipt_hash present;
    ndjson: full chain links)."""
    try:
        if path.suffix == ".ndjson":
            prev = None
            n = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n += 1
                if "prev_hash" in d:
                    if prev is not None and d["prev_hash"] != prev:
                        return False, f"chain break at record {n}"
                    prev = d.get("receipt_hash")
            return (True, f"{n} records chained") if n else (False, "empty")
        d = json.loads(path.read_text(encoding="utf-8"))
        if "receipt_hash" in d:
            return True, "receipt_hash present"
        return False, "no receipt_hash field"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def nr_full_entry(cfg, nr_id):
    """True iff a heading-level entry for nr_id exists in an NR file."""
    for f in cfg["nr_files"]:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        for m in NR_HEADING_STRICT.finditer(text):
            if m.group(1) == nr_id:
                return True, str(f.relative_to(FP))
    return False, None

def check_experiments_a(cfg, row, evidence, toks):
    exp_dir = cfg["exp_dir"]
    for eid in cfg["exp_ids"]:
        if re.search(r'\b' + eid + r'\b', evidence):
            script = cfg["script_for"][eid]
            p = exp_dir / script
            if not p.is_file():
                flag("a", row, "STALE",
                     f"experiment {eid}: script {script} missing on disk")
            receipt = resolve_receipt("a", cfg["receipt_for"][eid])
            ok, why = receipt_ok(receipt)
            if not ok:
                flag("a", row, "STALE",
                     f"experiment {eid}: receipt {cfg['receipt_for'][eid]} {why}")
    if re.search(r'identit(y|ies)|symmetry', evidence, re.I):
        p = exp_dir / cfg["script_for"]["IDENTITY"]
        if not p.is_file():
            flag("a", row, "STALE",
                 "identity symmetry: identity_symmetry_check.py missing")


def check_experiments_b(cfg, row, evidence, toks):
    for eid, (fidx, func) in cfg["func_for"].items():
        if re.search(r'\b' + re.escape(eid) + r'\b', evidence) or \
                f"EXP-AB-{eid}" in evidence:
            src = cfg["exp_files"][fidx]
            if not src.is_file():
                flag("b", row, "STALE",
                     f"experiment {eid}: source file {src.name} missing")
                continue
            if not re.search(r'def\s+' + func + r'\s*\(', src.read_text()):
                flag("b", row, "STALE",
                     f"experiment {eid}: function {func}() not found in {src.name}")
            receipt = resolve_receipt("b", f"EXP-AB-{eid}.json")
            ok, why = receipt_ok(receipt)
            if not ok:
                flag("b", row, "STALE",
                     f"experiment {eid}: receipt EXP-AB-{eid}.json {why}")
    if "repro_multiseed" in evidence:
        p = cfg["exp_files"][2]
        if not p.is_file():
            flag("b", row, "STALE", "repro_multiseed.py missing")


def check_receipts(arch, row, evidence, toks):
    path_tails = {p.split("/")[-1] for p in toks["receipt_paths"]}
    names = [n for n in toks["receipts"] if n not in path_tails]
    for name in names + toks["receipt_paths"]:
        # skip receipts already covered by per-experiment checks (they are
        # re-checked here too, harmless)
        path = resolve_receipt(arch, name)
        if path is None:
            # maybe the token is a sub-path like receipts/X.json at FP root
            alt = FP / name
            path = alt if alt.is_file() else None
        ok, why = receipt_ok(path)
        if not ok:
            flag(arch, row, "STALE",
                 f"receipt '{name}' {why} on disk")
        elif toks["hash_chained"]:
            okc, whyc = check_hash_chain(path)
            if not okc:
                flag(arch, row, "STALE",
                     f"receipt '{name}' claimed hash-chained: {whyc}")


def check_nrs(arch, row, evidence, toks):
    cfg = ARCHES[arch]
    for nr in toks["nrs"]:
        ok, where = nr_full_entry(cfg, nr)
        if not ok:
            flag(arch, row, "STALE",
                 f"cited {nr} has no full entry in any research/negative_results.md")


def identity_covered_exps(arch):
    """Experiment IDs with on-disk identity evidence (sha_before==sha_after
    or byte_identical:true in a receipt)."""
    covered = set()
    cfg = ARCHES[arch]
    rdir = cfg["receipts"]
    if not rdir.is_dir():
        return covered
    for p in rdir.glob("*.json"):
        try:
            txt = p.read_text(encoding="utf-8")
        except OSError:
            continue
        has_id = (re.search(r'"byte_identical"\s*:\s*true', txt) is not None
                  or re.search(r'"sha_before"\s*:\s*"([0-9a-f]+)"\s*,\s*'
                               r'"sha_after"\s*:\s*"\1"', txt) is not None)
        if not has_id:
            continue
        # which experiments does this receipt cover?
        if p.name == "k8_r1_wiring_confirmation.json":
            m = re.search(r'"b_(\w+)_reruns"\s*:\s*\{([^}]*)\}', txt, re.S)
            # k8 gate (b) covers K1-K3 explicitly
            covered |= {"K1", "K2", "K3"}
        if "post_wire_reruns" in txt:
            try:
                d = json.loads(txt)
                for k, v in d.get("post_wire_reruns", {}).items():
                    if isinstance(v, dict) and v.get("byte_identical"):
                        m2 = re.search(r'k(\d)', k)
                        if m2:
                            covered.add("K" + m2.group(1))
            except Exception:  # noqa: BLE001
                pass
    return covered


def check_identity_claims(arch, row, evidence, toks):
    """byte/bit-identical claims are scoped per clause: every experiment ID
    named in the same clause as the identity adjective must have identity
    evidence on disk."""
    if not (toks["byte_identical"] or toks["bit_identical"]):
        return
    covered = identity_covered_exps(arch)
    cfg = ARCHES[arch]
    clauses = re.split(r'[.;]', evidence)
    for clause in clauses:
        if not re.search(r'byte-identical|bit-identical', clause, re.I):
            continue
        ids = set(re.findall(r'\b(K[1-8])\b', clause))
        if arch == "b":
            ids |= set(re.findall(r'\b(C[1-4]B?|K3B|M1|BAR)\b', clause))
            ids |= {m.replace("EXP-AB-", "")
                    for m in EXP_FP_RE.findall(clause)
                    if m.startswith("EXP-AB-")}
        missing = sorted(i for i in ids if i not in covered)
        if missing:
            flag(arch, row, "OVERCLAIMED",
                 f"claims identity for {', '.join(missing)} in clause "
                 f"'{clause.strip()[:90]}...' but no receipt records "
                 "byte identity for them")


def check_level_justification(arch, row, level, evidence):
    low = evidence.lower()
    cfg = ARCHES[arch]
    if level == "REPRODUCED":
        toks = extract_tokens(evidence)
        repro = [n for n in toks["receipts"] + toks["receipt_paths"]
                 if n.split("/")[-1].startswith("repro_")]
        live = [n for n in repro if resolve_receipt(arch, n) is not None]
        if not live:
            flag(arch, row, "OVERCLAIMED",
                 "REPRODUCED requires a cited multi-seed reproduction "
                 "receipt (repro_*); none found on disk")
    elif level == "CAUSAL":
        lesion_sig = re.search(r'lesion|ablat|kill|frozen|freeze|disabl|no-memory',
                               low)
        cited_lesion = any(
            re.search(r'\b' + e + r'\b', evidence) for e in cfg["lesion_exps"])
        label_test = ("label" in low and "identical" in low)
        if not (lesion_sig or cited_lesion or label_test):
            flag(arch, row, "OVERCLAIMED",
                 "CAUSAL requires a cited lesion/ablation/kill experiment; "
                 "evidence cites none")
    elif level == "GENERALIZING":
        if not re.search(r'generali|noisy|multi-reversal|transfer|beyond|P2|P3',
                         evidence):
            flag(arch, row, "OVERCLAIMED",
                 "GENERALIZING requires cited generalization experiments; "
                 "none found")
    elif level == "ADAPTIVE":
        if not re.search(r'learn|adapt|gain_history|re-learn|feedback|moving',
                         low):
            flag(arch, row, "OVERCLAIMED",
                 "ADAPTIVE requires evidence of experience-driven change; "
                 "none found")
    elif level in ("INTEGRATED", "EXECUTED", "PRESENT"):
        toks = extract_tokens(evidence)
        if not (toks["receipts"] or toks["exps"] or toks["nrs"]
                or re.search(r'\b(K[1-8]|C[1-4]B?|K3B|M1|BAR)\b', evidence)):
            flag(arch, row, "WARN",
                 f"{level}: no verifiable citation token (receipt/experiment/NR) "
                 "in evidence text")


def check_row(arch, r):
    row = r["component"]
    level = base_level(r["level_raw"])
    if level is None:
        flag(arch, row, "WARN", f"unparseable level '{r['level_raw']}'")
        return
    toks = extract_tokens(r["evidence"])
    check_receipts(arch, row, r["evidence"], toks)
    check_nrs(arch, row, r["evidence"], toks)
    if arch == "a":
        check_experiments_a(ARCHES["a"], row, r["evidence"], toks)
    else:
        check_experiments_b(ARCHES["b"], row, r["evidence"], toks)
    check_identity_claims(arch, row, r["evidence"], toks)
    check_level_justification(arch, row, level, r["evidence"])

# ---- ledger-internal consistency checks ----

def repro_section_changes(text):
    """Parse 'Maturity changes from this reproduction' bullets:
    '- `comp`: OLD -> **NEW**' -> list of (fragments, new_level)."""
    out = []
    for line in text.splitlines():
        m = re.match(
            r'^\s*-\s+`?([^`:]+?)`?\s*:\s*[A-Z][A-Z, ()]*\s*→\s*\*\*([A-Z]+)\*\*',
            line)
        if m:
            frags = [f.strip().strip("`").lower()
                     for f in re.split(r'\+', m.group(1))]
            out.append((frags, m.group(2)))
    return out


def check_repro_consistency_a(rows):
    text = ARCHES["a"]["maturity"].read_text(encoding="utf-8")
    for frags, new_level in repro_section_changes(text):
        for r in rows:
            comp = r["component"].lower()
            if any(f and f in comp for f in frags):
                cur = base_level(r["level_raw"])
                if cur != new_level:
                    direction = ("under-promoted" if RANK[cur] < RANK[new_level]
                                 else "over-promoted")
                    flag("a", r["component"], "STALE",
                         f"table claims {cur} but the REPRODUCIBILITY section "
                         f"assigns {new_level} ({direction}); sync the table")


def check_repro_consistency_b(rows):
    text = ARCHES["b"]["maturity"].read_text(encoding="utf-8")
    for frags, new_level in repro_section_changes(text):
        for r in rows:
            comp = r["component"].lower()
            if any(f and f in comp for f in frags):
                cur = base_level(r["level_raw"])
                if cur != new_level:
                    flag("b", r["component"], "STALE",
                         f"table claims {cur} but the REPRODUCIBILITY section "
                         f"assigns {new_level}; sync the table")


def parse_md_table(text, header_re):
    """Generic markdown-table parse after a header matching header_re."""
    rows, in_t = [], False
    for line in text.splitlines():
        if header_re.search(line):
            in_t = True
            continue
        if in_t:
            if not line.strip():
                continue
            if re.match(r'^\|\s*-+', line):
                continue
            if not line.startswith("|"):
                break
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def check_b_kill_table():
    text = ARCHES["b"]["maturity"].read_text(encoding="utf-8")
    kill = parse_md_table(text, re.compile(r'^## Kill-experiment outcomes'))
    repl = parse_md_table(text, re.compile(r'^\| Experiment \|'))
    # replication outcome -> keyword the kill table must now reflect
    expect = {
        "K1": ["KILL"], "K3": ["KILL"], "K5": ["KILL"],
        "C1": ["MIXED"], "C2": ["KILL"], "C4": ["KILL"], "BAR": ["MIXED"],
    }
    kill_verdict = {}
    for cols in kill:
        if len(cols) >= 3:
            kill_verdict[cols[0].split()[0].upper()] = cols[2]
    replicated = set()
    for cols in repl:
        if len(cols) >= 5:
            name = cols[0].split()[0].upper()
            replicated.add(name)
    for name, kws in expect.items():
        if name in replicated:
            v = kill_verdict.get(name, "")
            if not all(k.upper() in v.upper() for k in kws):
                flag("b", f"kill-table {name}", "STALE",
                     f"kill-table verdict '{v}' does not reflect the "
                     f"replication outcome (expects keyword(s) {kws}); "
                     "annotate with the 2026-10-07 replication result")


def check_b_open_gaps():
    text = ARCHES["b"]["maturity"].read_text(encoding="utf-8")
    m = re.search(r'## Open gaps.*?(?=\n## |\Z)', text, re.S)
    repro_exists = any(ARCHES["b"]["receipts"].glob("repro_EXP-AB-*.json"))
    if m and repro_exists and "single-seed" in m.group(0):
        flag("b", "open gaps #1", "STALE",
             "open gap #1 still says Phase-3 battery is single-seed, but "
             "repro_EXP-AB-*.json receipts exist on disk; reword with the "
             "mixed multi-seed outcome")


def check_k8_reject_recorded():
    """K8's own script mandates a negative-result entry on REJECT."""
    k8 = ARCHES["a"]["receipts"] / "k8_r1_wiring_confirmation.json"
    try:
        d = json.loads(k8.read_text(encoding="utf-8"))
    except OSError:
        flag("a", "K8", "STALE", "k8_r1_wiring_confirmation.json missing")
        return
    if d.get("decision") == "REJECT":
        found = False
        for f in ARCHES["a"]["nr_files"]:
            if f.is_file() and re.search(r'(?m)^#{1,4}\s.*(K8|NR-A-010)',
                                         f.read_text(encoding="utf-8")):
                found = True
        if not found:
            flag("a", "K8 REJECT", "STALE",
                 "K8 decision is REJECT (gate a1 failed) but no "
                 "negative-result entry records it; the K8 script mandates "
                 "recording a REJECT in research/negative_results.md")


# ---- report ----

def main():
    rows_by_arch = {}
    for arch in ("a", "b"):
        rows = parse_component_table(ARCHES[arch]["maturity"])
        rows_by_arch[arch] = rows
        for r in rows:
            check_row(arch, r)
    check_repro_consistency_a(rows_by_arch["a"])
    check_repro_consistency_b(rows_by_arch["b"])
    check_b_kill_table()
    check_b_open_gaps()
    check_k8_reject_recorded()

    n_stale = sum(1 for f in findings if f["type"] == "STALE")
    n_over = sum(1 for f in findings if f["type"] == "OVERCLAIMED")
    n_warn = sum(1 for f in findings if f["type"] == "WARN")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ["prototypes/architecture-a/MATURITY.md",
                  "prototypes/architecture-b/MATURITY.md"],
        "rows": {arch: [{"component": r["component"],
                         "level": base_level(r["level_raw"]),
                         "level_raw": r["level_raw"]}
                        for r in rows_by_arch[arch]]
                 for arch in ("a", "b")},
        "findings": findings,
        "summary": {
            "rows_checked": {arch: len(rows_by_arch[arch])
                             for arch in ("a", "b")},
            "STALE": n_stale, "OVERCLAIMED": n_over, "WARN": n_warn,
            "verdict": "GREEN" if (n_stale == 0 and n_over == 0) else "RED",
        },
    }
    out = Path(__file__).resolve().parent / "maturity_audit_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 70)
    print("MATURITY AUDIT — Flesh Pits architectures A/B")
    print("=" * 70)
    for arch in ("a", "b"):
        fa = [f for f in findings if f["arch"] == arch]
        print(f"\n[arch-{arch}] rows checked: {len(rows_by_arch[arch])}")
        print(f"  STALE: {sum(1 for f in fa if f['type']=='STALE')}  "
              f"OVERCLAIMED: {sum(1 for f in fa if f['type']=='OVERCLAIMED')}  "
              f"WARN: {sum(1 for f in fa if f['type']=='WARN')}")
        for f in fa:
            print(f"  [{f['type']}] {f['row']}\n      {f['message']}")
    cross = [f for f in findings if f["arch"] not in ("a", "b")]
    for f in cross:
        print(f"  [{f['type']}] {f['row']}\n      {f['message']}")
    print("\n" + "=" * 70)
    print(f"VERDICT: {report['summary']['verdict']} "
          f"(STALE={n_stale} OVERCLAIMED={n_over} WARN={n_warn})")
    print(f"report: {out}")
    return 0 if report["summary"]["verdict"] == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
