"""
app.py  —  Ayurveda NLP Pipeline · Streamlit UI  [UPGRADED v2]
===============================================================
PRC Pramana Interns 2026
New in v2:
  - NLP-08 tab: Consolidated NER + Relations with quality scores
  - Confidence breakdown charts in Relations tab
  - Entity quality flags visible in NER tab
  - Health check panel
Run:  streamlit run app.py
"""

import io
import json
import time
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Ayurveda NLP Pipeline",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0d1f1a; border-right: 1px solid #1e3b30; }
section[data-testid="stSidebar"] * { color: #c8ddd4 !important; }
.main { background: #f5f0e8; }
.block-container { padding: 2rem 2.5rem; max-width: 1300px; }
.pipeline-header {
    background: linear-gradient(135deg, #0d2b1f 0%, #1a4d35 60%, #0f3527 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
}
.pipeline-header h1 { font-family: 'Playfair Display', serif; color: #c8e6c9; font-size: 2.2rem; margin: 0 0 0.4rem 0; }
.pipeline-header p { color: #7fb899; margin: 0; font-size: 0.9rem; font-weight: 300; }
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card { background: white; border-radius: 12px; padding: 1.2rem 1.5rem; flex: 1; min-width: 130px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07); border-left: 4px solid #2d7a4f; }
.stat-card .val { font-family: 'Playfair Display', serif; font-size: 2rem; color: #1a3d2b; line-height: 1; }
.stat-card .lbl { font-size: 0.75rem; color: #7a9b8a; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }
.stat-card.warn { border-left-color: #e07b00; }
.stat-card.good { border-left-color: #2d7a4f; }
.ent-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
    font-family: 'DM Mono', monospace; margin-right: 4px; }
.ent-HERB       { background: #c8f5d8; color: #1a5c30; }
.ent-DISEASE    { background: #fdd; color: #8b0000; }
.ent-DOSHA      { background: #d0e8ff; color: #003a7d; }
.ent-PROCEDURE  { background: #fff0c8; color: #7a4f00; }
.ent-SOURCE_REF { background: #ece8ff; color: #3d2a8a; }
.ent-BODY_PART  { background: #ffecd0; color: #7a3800; }
.ent-PROPERTY   { background: #e0f7fa; color: #006064; }
.ent-QUANTITY   { background: #fce4ec; color: #880e4f; }
.ent-PLANT_PART { background: #f1f8e9; color: #33691e; }
.ent-INGREDIENT { background: #fbe9e7; color: #bf360c; }
.ent-FORMULATION{ background: #e8eaf6; color: #283593; }
.ent-DIET       { background: #f9fbe7; color: #558b2f; }
.rel-card { background: white; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e8ede9; }
.rel-subj { font-weight: 600; color: #1a3d2b; }
.rel-type { background: #0d2b1f; color: #7fb899; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-family: 'DM Mono', monospace; }
.rel-obj  { font-weight: 500; color: #2d5a3d; }
.rel-conf-high   { border-left: 3px solid #2d7a4f; }
.rel-conf-medium { border-left: 3px solid #f0a500; }
.rel-conf-low    { border-left: 3px solid #aaa; }
.section-title { font-family: 'Playfair Display', serif; color: #1a3d2b; font-size: 1.35rem;
    border-bottom: 2px solid #c8ddd4; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
.log-box { background: #0d1f1a; color: #7fb899; font-family: 'DM Mono', monospace; font-size: 0.75rem;
    padding: 1rem; border-radius: 8px; max-height: 280px; overflow-y: auto; line-height: 1.6; }
.health-ok   { background: #d4edda; border-left: 4px solid #2d7a4f; border-radius: 8px; padding: 0.75rem 1rem; }
.health-warn { background: #fff3cd; border-left: 4px solid #e07b00; border-radius: 8px; padding: 0.75rem 1rem; }
.flag-chip { background: #ffecd0; color: #7a3800; padding: 1px 7px; border-radius: 12px; font-size: 0.68rem; font-family: 'DM Mono', monospace; margin-right: 3px; }
div[data-testid="stFileUploader"] { border: 2px dashed #7fb899 !important; border-radius: 12px !important; background: #f0f7f2 !important; }
.stButton > button { background: #1a4d35; color: #c8e6c9; border: none; border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-weight: 500; padding: 0.55rem 1.8rem; font-size: 0.9rem; }
.stButton > button:hover { background: #2d7a4f; transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #c8ddd4; }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; color: #7a9b8a; font-weight: 500;
    padding: 0.5rem 1.2rem; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: #1a4d35 !important; color: #c8e6c9 !important; }
div[data-testid="stDownloadButton"] > button { background: #2d5a3d !important; color: white !important; border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
for key, default in {
    "pipeline_done": False,
    "output_dir": None,
    "stage_results": {},
    "log_lines": [],
    "run_time": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

LABEL_COLORS = {
    "HERB": "#c8f5d8", "DISEASE": "#fdd", "DOSHA": "#d0e8ff",
    "PROCEDURE": "#fff0c8", "SOURCE_REF": "#ece8ff", "BODY_PART": "#ffecd0",
    "PROPERTY": "#e0f7fa", "QUANTITY": "#fce4ec", "PLANT_PART": "#f1f8e9",
    "INGREDIENT": "#fbe9e7", "FORMULATION": "#e8eaf6", "DIET": "#f9fbe7",
}

TASK_META = {
    1: ("NLP-01", "Text Cleaning"),
    2: ("NLP-02", "Language Detection"),
    3: ("NLP-03", "Annotation"),
    5: ("NLP-05", "NER Training"),
    6: ("NLP-06", "Semantic Tagging"),
    7: ("NLP-07", "Relationship Mining"),
    8: ("NLP-08", "Consolidation"),       # NEW
}
PIPELINE_ORDER = [1, 2, 3, 5, 6, 7, 8]


# ── Pipeline runner ───────────────────────────────────────────────────
def run_pipeline_streamlit(uploaded_files, tasks, lang_hint, log_list, output_dir: Path):
    from utils.logger import get_streamlit_logger
    from utils.report import PipelineReport
    from nlp01_text_cleaning    import TextCleaner
    from nlp02_language_detect  import LanguageDetector
    from nlp03_annotation       import AnnotationGuidelineApplier
    from nlp05_ner              import NERModelTrainer
    from nlp06_semantic_tagging import SemanticTagger
    from nlp07_relationship     import RelationshipExtractor
    from nlp08_consolidation    import NLPConsolidator

    TASK_CLASSES = {
        1: TextCleaner, 2: LanguageDetector, 3: AnnotationGuidelineApplier,
        5: NERModelTrainer, 6: SemanticTagger, 7: RelationshipExtractor,
        8: NLPConsolidator,
    }

    logger = get_streamlit_logger(log_list, verbose=True)
    report = PipelineReport(output_dir)
    stage_results = {}

    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_files = []
    for uf in uploaded_files:
        p = input_dir / uf.name
        p.write_bytes(uf.read())
        input_files.append(p)
    log_list.append(f"[INFO] {len(input_files)} input file(s) ready")

    current_inputs = input_files
    task_sequence  = [t for t in PIPELINE_ORDER if t in tasks]

    for task_num in task_sequence:
        task_id, task_name = TASK_META[task_num]
        TaskClass = TASK_CLASSES[task_num]
        log_list.append(f"\n[INFO] ── Running {task_id}: {task_name} ──")

        task_out_dir = output_dir / f"{task_id.lower().replace('-','_')}_{task_name.lower().replace(' ','_')}"
        task_out_dir.mkdir(parents=True, exist_ok=True)

        task = TaskClass(
            input_files=current_inputs,
            output_dir=task_out_dir,
            logger=logger,
            lang_hint=lang_hint,
            prev_outputs=stage_results,
        )
        t0 = time.time()
        try:
            results = task.run()
        except Exception as e:
            log_list.append(f"[ERROR] {task_id} failed: {e}")
            results = {"task": task_id, "output_files": current_inputs, "summary": f"ERROR: {e}"}
        elapsed = time.time() - t0

        stage_results[task_num] = results
        report.add_stage(task_id, task_name, results, elapsed)
        if results.get("output_files"):
            current_inputs = [f for f in results["output_files"] if hasattr(f, "suffix")]
        log_list.append(f"[INFO] ✓ {task_id} done in {elapsed:.1f}s — {results.get('summary','')}")

    report.save()
    return stage_results


# ── Data loaders ─────────────────────────────────────────────────────
def load_all_entities(output_dir: Path) -> list:
    entities = []
    seen = set()
    for pattern in ["*_annotated.json", "*_ner.json"]:
        for f in output_dir.rglob(pattern):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                items = data.get("annotations", data.get("entities", []))
                src = data.get("source_file", f.name)
                for e in items:
                    key = (e["text"].strip().lower(), e.get("label",""))
                    if key not in seen:
                        seen.add(key)
                        entities.append({**e, "source_file": src})
            except Exception:
                pass
    return entities


def load_all_relations(output_dir: Path) -> list:
    relations = []
    seen = set()
    for f in output_dir.rglob("*_relations.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            src = data.get("source_file", f.name)
            for r in data.get("relations", []):
                key = (r["relation"], r.get("subject","").lower(), r.get("object","").lower())
                if key not in seen:
                    seen.add(key)
                    relations.append({**r, "source_file": src})
        except Exception:
            pass
    return relations


def load_lang_results(output_dir: Path) -> list:
    results = []
    for f in output_dir.rglob("*_lang.json"):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results


def load_nlp08_health(output_dir: Path) -> dict:
    for f in output_dir.rglob("nlp08_health.json"):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_consolidated_ner(output_dir: Path):
    for f in output_dir.rglob("ner_consolidated.xlsx"):
        try:
            import pandas as pd
            return pd.read_excel(f)
        except Exception:
            pass
    return None


def load_consolidated_relations(output_dir: Path):
    for f in output_dir.rglob("relations_consolidated.csv"):
        try:
            import pandas as pd
            return pd.read_csv(f)
        except Exception:
            pass
    return None


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0; border-bottom: 1px solid #1e3b30; margin-bottom: 1rem;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.1rem; color: #c8e6c9; font-weight: 700;'>🌿 Ayurveda NLP</div>
        <div style='font-size: 0.72rem; color: #4a7a5e; margin-top: 2px;'>PRC Pramana · Interns 2026 · v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.75rem;color:#8aad9e;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Pipeline Tasks</div>', unsafe_allow_html=True)
    tasks_selected = []
    task_labels = {
        1: "NLP-01 · Text Cleaning",
        2: "NLP-02 · Language Detection",
        3: "NLP-03 · Annotation",
        5: "NLP-05 · NER Training",
        6: "NLP-06 · Semantic Tagging",
        7: "NLP-07 · Relationship Mining",
        8: "NLP-08 · Consolidation ✨",
    }
    for num, label in task_labels.items():
        if st.checkbox(label, value=True, key=f"task_{num}"):
            tasks_selected.append(num)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    lang_hint = st.selectbox(
        "Language Hint", ["auto", "sanskrit", "malayalam", "english", "mixed"]
    )
    st.markdown("""
    <div style='font-size:0.72rem;color:#4a7a5e;padding:0.75rem;background:#0a1812;border-radius:8px;line-height:1.7;margin-top:1rem;'>
        <b style='color:#7fb899'>Flow</b><br>
        OCR .txt → Clean → Detect → Annotate → NER → Semantic → Relations → <b style='color:#c8e6c9'>Consolidate</b> → QA
    </div>
    """, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-header">
    <h1>Ayurveda NLP Pipeline</h1>
    <p>OCR text → Entities → Relations → Consolidated QA Export · Sanskrit · Malayalam · English · v2</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab_run, tab_ner, tab_rel, tab_nlp8, tab_qa = st.tabs([
    "▶  Run Pipeline", "🔬  NER Entities", "🔗  Relations", "🧩  NLP-08 Consolidated", "📤  QA Export"
])


# ════════════════════════════════════════════════════════════════════ #
# TAB 1 — RUN PIPELINE
# ════════════════════════════════════════════════════════════════════ #
with tab_run:
    st.markdown('<div class="section-title">Upload OCR Text Files</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your OCR .txt files here",
        type=["txt"], accept_multiple_files=True,
    )
    if uploaded:
        st.success(f"✅  {len(uploaded)} file(s) ready: {', '.join(f.name for f in uploaded)}")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀  Run Pipeline", disabled=not uploaded or not tasks_selected)

    if run_btn and uploaded:
        output_dir = Path(tempfile.mkdtemp(prefix="ayurveda_nlp_"))
        st.session_state.update({
            "output_dir": str(output_dir),
            "log_lines": [],
            "pipeline_done": False,
        })
        t_start = time.time()
        with st.spinner("Running pipeline…"):
            log_lines = st.session_state["log_lines"]
            stage_results = run_pipeline_streamlit(
                uploaded_files=uploaded, tasks=tasks_selected,
                lang_hint=lang_hint, log_list=log_lines, output_dir=output_dir,
            )
            st.session_state["stage_results"] = stage_results
            st.session_state["run_time"]      = round(time.time() - t_start, 1)
            st.session_state["pipeline_done"] = True
        st.success(f"✅  Pipeline completed in {st.session_state['run_time']}s — explore tabs above.")

    if st.session_state["pipeline_done"] and st.session_state["stage_results"]:
        output_dir = Path(st.session_state["output_dir"])
        results    = st.session_state["stage_results"]

        st.markdown('<div class="section-title">Pipeline Summary</div>', unsafe_allow_html=True)
        total_entities = sum(
            r.get("entity_total", 0) or sum(r.get("entity_counts", {}).values())
            for r in results.values()
        )
        total_relations = sum(r.get("total_relations", 0) for r in results.values())
        files_processed = max(
            (len(r.get("output_files", [])) for r in results.values()), default=0
        )
        health = load_nlp08_health(output_dir)
        status_class = "good" if health.get("status") == "OK" else "warn"

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card"><div class="val">{files_processed}</div><div class="lbl">Files</div></div>
            <div class="stat-card"><div class="val">{total_entities}</div><div class="lbl">Entities</div></div>
            <div class="stat-card"><div class="val">{total_relations}</div><div class="lbl">Relations</div></div>
            <div class="stat-card"><div class="val">{st.session_state['run_time']}s</div><div class="lbl">Run time</div></div>
            <div class="stat-card {status_class}"><div class="val">{health.get('status','—')}</div><div class="lbl">Health</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Health issues
        if health.get("issues"):
            st.markdown("**Pipeline Health:**")
            for issue in health["issues"]:
                if "WARNING" in issue:
                    st.warning(issue)
                else:
                    st.info(issue)

        for task_num in PIPELINE_ORDER:
            if task_num not in results:
                continue
            tid, tname = TASK_META[task_num]
            r = results[task_num]
            with st.expander(f"**{tid}** · {tname}", expanded=False):
                st.code(r.get("summary", "—"))
                clean = {k: v for k, v in r.items() if k not in ("output_files", "task") and not isinstance(v, list)}
                if clean:
                    st.json(clean)

        st.markdown('<div class="section-title">Run Log</div>', unsafe_allow_html=True)
        log_text = "\n".join(st.session_state["log_lines"])
        st.markdown(f'<div class="log-box"><pre>{log_text}</pre></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════ #
# TAB 2 — NER ENTITIES
# ════════════════════════════════════════════════════════════════════ #
with tab_ner:
    if not st.session_state["pipeline_done"]:
        st.info("Run the pipeline first (▶ Run Pipeline tab).")
    else:
        output_dir = Path(st.session_state["output_dir"])
        entities   = load_all_entities(output_dir)
        st.markdown('<div class="section-title">Named Entity Recognition</div>', unsafe_allow_html=True)

        if not entities:
            st.warning("No entities found. Make sure NLP-03 / NLP-05 ran successfully.")
        else:
            all_labels = sorted(set(e.get("label","") for e in entities))
            all_files  = sorted(set(e.get("source_file","") for e in entities))

            col_f, col_l, col_s = st.columns([2, 2, 2])
            with col_f:
                filter_file  = st.multiselect("Source file", all_files, default=all_files, key="ner_file")
            with col_l:
                filter_label = st.multiselect("Label", all_labels, default=all_labels, key="ner_label")
            with col_s:
                search_term = st.text_input("Search entity text", "", key="ner_search")

            filtered = [
                e for e in entities
                if e.get("source_file","") in filter_file
                and e.get("label","") in filter_label
                and (not search_term or search_term.lower() in e["text"].lower())
            ]

            label_counts = {}
            for e in filtered:
                lbl = e.get("label", "OTHER")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            badges = " ".join(
                f'<span class="ent-badge ent-{lbl}">{lbl} <b>{cnt}</b></span>'
                for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
            )
            st.markdown(f'<div style="margin:0.75rem 0 1rem 0">{badges}</div>', unsafe_allow_html=True)
            st.caption(f"Showing {len(filtered)} of {len(entities)} entities")

            try:
                import pandas as pd
                df = pd.DataFrame([
                    {"Entity": e["text"], "Label": e.get("label",""), "Source File": e.get("source_file","")}
                    for e in filtered
                ])
                def color_label(val):
                    bg = LABEL_COLORS.get(val, "#eee")
                    return f"background-color: {bg}; font-weight: 600; font-size: 0.78rem;"
                st.dataframe(df.style.applymap(color_label, subset=["Label"]), use_container_width=True, height=460)
            except Exception:
                for e in filtered[:200]:
                    lbl = e.get("label","")
                    st.markdown(f'<span class="ent-badge ent-{lbl}">{lbl}</span> {e["text"]}', unsafe_allow_html=True)

            lang_data = load_lang_results(output_dir)
            if lang_data:
                st.markdown('<div class="section-title">Language Detection</div>', unsafe_allow_html=True)
                for ld in lang_data:
                    pct = ld.get("detection", {}).get("percentages", {})
                    dominant = ld.get("detection", {}).get("dominant", "?")
                    mixed    = ld.get("detection", {}).get("is_mixed", False)
                    st.markdown(f"**{ld.get('source_file','')}** — dominant: `{dominant}` {'*(mixed)*' if mixed else ''}")
                    cols = st.columns(3)
                    for i, (lang, pct_val) in enumerate(pct.items()):
                        cols[i].metric(lang.capitalize(), f"{pct_val}%")


# ════════════════════════════════════════════════════════════════════ #
# TAB 3 — RELATIONS
# ════════════════════════════════════════════════════════════════════ #
with tab_rel:
    if not st.session_state["pipeline_done"]:
        st.info("Run the pipeline first (▶ Run Pipeline tab).")
    else:
        output_dir = Path(st.session_state["output_dir"])
        relations  = load_all_relations(output_dir)
        st.markdown('<div class="section-title">Relationship Mining</div>', unsafe_allow_html=True)

        if not relations:
            st.warning("No relations found. Make sure NLP-07 ran.")
        else:
            # Confidence breakdown
            conf_counts = {"high": 0, "medium": 0, "low": 0}
            for r in relations:
                conf_counts[r.get("confidence","low")] = conf_counts.get(r.get("confidence","low"), 0) + 1

            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 High confidence", conf_counts["high"])
            c2.metric("🟡 Medium confidence", conf_counts["medium"])
            c3.metric("⚪ Low confidence", conf_counts["low"])

            all_rel_types = sorted(set(r.get("relation","") for r in relations))
            all_confs     = sorted(set(r.get("confidence","") for r in relations))

            col_rt, col_cf = st.columns([3, 2])
            with col_rt:
                filter_rel  = st.multiselect("Relation type", all_rel_types,
                    default=[t for t in all_rel_types if t != "CO_OCCURS_WITH"], key="rel_type")
            with col_cf:
                filter_conf = st.multiselect("Confidence", all_confs, default=all_confs, key="rel_conf")

            search_rel = st.text_input("Search subject or object", "", key="rel_search")

            filtered_rels = [
                r for r in relations
                if r.get("relation","") in filter_rel
                and r.get("confidence","") in filter_conf
                and (not search_rel or
                     search_rel.lower() in r.get("subject","").lower() or
                     search_rel.lower() in r.get("object","").lower())
            ]

            rel_counts = {}
            for r in filtered_rels:
                rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1

            badges = " ".join(
                f'<span style="background:#e8f5e9;color:#1b5e20;padding:2px 10px;border-radius:20px;font-size:0.75rem;margin:2px;">{rt} ({cnt})</span>'
                for rt, cnt in sorted(rel_counts.items(), key=lambda x: -x[1])
            )
            st.markdown(f'<div style="margin:0.5rem 0 1rem 0">{badges}</div>', unsafe_allow_html=True)
            st.caption(f"Showing {len(filtered_rels)} of {len(relations)} relations")

            for r in filtered_rels[:200]:
                conf = r.get("confidence", "medium")
                ctx  = (r.get("context","")[:80] + "…") if r.get("context","") else ""
                subj_lbl = r.get("subject_label","")
                obj_lbl  = r.get("object_label","")
                sb = f'<span class="ent-badge ent-{subj_lbl}">{subj_lbl}</span>' if subj_lbl else ""
                ob = f'<span class="ent-badge ent-{obj_lbl}">{obj_lbl}</span>' if obj_lbl else ""
                st.markdown(f"""
                <div class="rel-card rel-conf-{conf}">
                    <span class="rel-subj">{r.get("subject","")}</span> {sb}
                    <span class="rel-type">{r.get("relation","")}</span>
                    {ob} <span class="rel-obj">{r.get("object","")}</span>
                    <span style="margin-left:auto;font-size:0.7rem;color:#aaa">{conf}</span>
                </div>
                {'<div style="font-size:0.72rem;color:#999;margin:-0.4rem 0 0.6rem 1.2rem;font-style:italic">' + ctx + '</div>' if ctx else ''}
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════ #
# TAB 4 — NLP-08 CONSOLIDATED  (NEW)
# ════════════════════════════════════════════════════════════════════ #
with tab_nlp8:
    if not st.session_state["pipeline_done"]:
        st.info("Run the pipeline first (▶ Run Pipeline tab). Make sure NLP-08 · Consolidation is checked.")
    else:
        output_dir = Path(st.session_state["output_dir"])
        health     = load_nlp08_health(output_dir)
        cons_ner   = load_consolidated_ner(output_dir)
        cons_rel   = load_consolidated_relations(output_dir)

        st.markdown('<div class="section-title">NLP-08 — Consolidated NER Output</div>', unsafe_allow_html=True)

        # Health panel
        if health:
            status = health.get("status", "UNKNOWN")
            cls = "health-ok" if status == "OK" else "health-warn"
            issues_html = "".join(f"<div>• {i}</div>" for i in health.get("issues", []))
            st.markdown(f"""
            <div class="{cls}" style="margin-bottom:1.5rem;">
                <b>Pipeline Health: {status}</b><br>
                {issues_html if issues_html else "No issues detected."}
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Entities", health.get("total_entities", 0))
            c2.metric("Total Relations", health.get("total_relations", 0))
            c3.metric("Need Review", health.get("entities_need_review", 0))
            rel_conf = health.get("relation_confidence", {})
            c4.metric("High-conf Relations", rel_conf.get("high", 0))

        # Consolidated NER table
        if cons_ner is not None and not cons_ner.empty:
            st.markdown("**Consolidated NER — all entities, quality-scored**")
            st.caption(f"{len(cons_ner)} unique entities | sorted by label → quality score")

            col_lbl, col_flag = st.columns([3, 2])
            with col_lbl:
                all_labels = sorted(cons_ner["Label"].unique()) if "Label" in cons_ner.columns else []
                filt_labels = st.multiselect("Filter label", all_labels, default=all_labels, key="c8_label")
            with col_flag:
                show_review = st.checkbox("Show only entities needing review", value=False, key="c8_review")

            display_df = cons_ner.copy()
            if filt_labels and "Label" in display_df.columns:
                display_df = display_df[display_df["Label"].isin(filt_labels)]
            if show_review and "Needs_Review" in display_df.columns:
                display_df = display_df[display_df["Needs_Review"] == True]

            def color_label(val):
                bg = LABEL_COLORS.get(val, "#eee")
                return f"background-color: {bg}; font-weight: 600;"

            def color_quality(val):
                try:
                    v = float(val)
                    if v >= 70: return "background-color: #d4edda; color: #155724;"
                    if v >= 50: return "background-color: #fff3cd; color: #856404;"
                    return "background-color: #f8d7da; color: #721c24;"
                except Exception:
                    return ""

            style_cols = {}
            if "Label" in display_df.columns:
                style_cols["Label"] = color_label
            if "Quality_Score" in display_df.columns:
                style_cols["Quality_Score"] = color_quality

            styled = display_df.style
            for col, fn in style_cols.items():
                styled = styled.applymap(fn, subset=[col])

            st.dataframe(styled, use_container_width=True, height=400)

            # Download
            try:
                import io as _io
                buf = _io.BytesIO()
                display_df.to_excel(buf, index=False)
                st.download_button(
                    "⬇  Download filtered NER (Excel)",
                    data=buf.getvalue(),
                    file_name="ner_consolidated_filtered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                pass
        else:
            st.info("No consolidated NER data yet. Run with NLP-08 checked.")

        # Consolidated Relations
        st.markdown('<div class="section-title">Consolidated Relations</div>', unsafe_allow_html=True)
        if cons_rel is not None and not cons_rel.empty:
            st.caption(f"{len(cons_rel)} unique relations | sorted by confidence")
            if "Confidence" in cons_rel.columns:
                c1, c2, c3 = st.columns(3)
                c1.metric("High", int((cons_rel["Confidence"] == "high").sum()))
                c2.metric("Medium", int((cons_rel["Confidence"] == "medium").sum()))
                c3.metric("Low", int((cons_rel["Confidence"] == "low").sum()))
            st.dataframe(cons_rel, use_container_width=True, height=350)
            st.download_button(
                "⬇  Download relations_consolidated.csv",
                data=cons_rel.to_csv(index=False).encode(),
                file_name="relations_consolidated.csv",
                mime="text/csv",
            )
        else:
            st.info("No consolidated relations yet.")


# ════════════════════════════════════════════════════════════════════ #
# TAB 5 — QA EXPORT
# ════════════════════════════════════════════════════════════════════ #
with tab_qa:
    if not st.session_state["pipeline_done"]:
        st.info("Run the pipeline first (▶ Run Pipeline tab).")
    else:
        output_dir = Path(st.session_state["output_dir"])
        qa_dir     = output_dir / "qa_exports"

        st.markdown('<div class="section-title">QA Export Files</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#f0f7f2;border-radius:10px;padding:1rem 1.5rem;border-left:4px solid #2d7a4f;margin-bottom:1.5rem;">
            <b>How to use these files:</b><br>
            • <b>ner_output.xlsx</b> → <b>Harini</b> — QA Streamlit tool → Tab 2 (NER Validator)<br>
            • <b>relations_output.csv</b> → <b>Neha</b> — QA Streamlit tool → Tab 3 & 4 (Relationship)<br>
            • Both now include <b>Quality_Score</b> and <b>Flags</b> columns — focus review on flagged rows first.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📊 NER Output (Excel) — for Harini")
            ner_file = qa_dir / "ner_output.xlsx"
            if ner_file.exists():
                st.download_button("⬇  Download ner_output.xlsx", data=ner_file.read_bytes(),
                    file_name="ner_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                try:
                    import pandas as pd
                    df_ner = pd.read_excel(ner_file)
                    st.caption(f"{len(df_ner)} entities")
                    st.dataframe(df_ner.head(30), use_container_width=True)
                except Exception as e:
                    st.warning(f"Preview error: {e}")
            else:
                st.warning("ner_output.xlsx not found — run pipeline with NLP-08 checked.")

        with col_b:
            st.markdown("#### 🔗 Relations Output (CSV) — for Neha")
            rel_file = qa_dir / "relations_output.csv"
            if rel_file.exists():
                st.download_button("⬇  Download relations_output.csv", data=rel_file.read_bytes(),
                    file_name="relations_output.csv", mime="text/csv")
                try:
                    import pandas as pd
                    df_rel = pd.read_csv(rel_file)
                    st.caption(f"{len(df_rel)} relations")
                    st.dataframe(df_rel.head(30), use_container_width=True)
                except Exception as e:
                    st.warning(f"Preview error: {e}")
            else:
                st.warning("relations_output.csv not found.")

        report_file = output_dir / "pipeline_report.json"
        if report_file.exists():
            st.markdown('<div class="section-title">Pipeline Report</div>', unsafe_allow_html=True)
            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button("⬇  pipeline_report.json", data=report_file.read_bytes(),
                    file_name="pipeline_report.json", mime="application/json")
            with st.expander("View full report"):
                st.json(json.loads(report_file.read_text()))