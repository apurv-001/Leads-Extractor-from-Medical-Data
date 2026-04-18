"""
app.py  —  Physiotherapy Lead Extractor  (Streamlit UI)
Calls physio_ner.py for all NLP logic.
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from physio_ner import extract_physio_leads, process_batch, PHYSIO_KEYWORDS

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Physio Lead Extractor", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f0f4f8; }

[data-testid="stSidebar"] { background-color: #1a365d !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea { background-color: #243f6a !important; border-color: #3a5a8a !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background-color: #3a5a8a !important; }
[data-testid="stSidebar"] hr { border-color: #3a5a8a !important; }

.main-header {
    background: linear-gradient(135deg, #1a365d 0%, #2d6a9f 100%);
    color: white; padding: 1.8rem 2.5rem; border-radius: 16px;
    margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(26,54,93,0.3);
}
.main-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
.main-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.9rem; }

.card { background:white; border-radius:14px; padding:1.5rem;
        box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:1rem; }

.stat-box { background:white; border-radius:12px; padding:1.2rem 1rem;
            text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
.stat-num   { font-size:2rem; font-weight:700; color:#1a365d; }
.stat-label { font-size:0.78rem; color:#6b7280; margin-top:2px; }

.result-physio  { border-left:4px solid #16a34a; background:#f0fdf4;
                  padding:1rem 1.2rem; border-radius:0 10px 10px 0; margin-bottom:0.8rem; }
.result-none    { border-left:4px solid #94a3b8; background:#f8fafc;
                  padding:1rem 1.2rem; border-radius:0 10px 10px 0; margin-bottom:0.8rem; }
.label-physio   { background:#16a34a; color:white; padding:3px 12px;
                  border-radius:20px; font-size:0.78rem; font-weight:600; }
.label-none     { background:#94a3b8; color:white; padding:3px 12px;
                  border-radius:20px; font-size:0.78rem; font-weight:600; }
.kw-chip        { display:inline-block; background:#dbeafe; color:#1e40af;
                  padding:2px 10px; border-radius:12px; font-size:0.75rem;
                  font-weight:500; margin:2px; }
.conf-bar-wrap  { background:#e5e7eb; border-radius:6px; height:8px; width:100%; margin-top:4px; }
.conf-bar       { border-radius:6px; height:8px; }

.stTabs [data-baseweb="tab-list"] { background:white; border-radius:12px;
    padding:6px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:1rem; gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:8px !important; font-weight:500; padding:8px 20px; }
.stTabs [aria-selected="true"] { background-color:#1a365d !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Physiotherapy Lead Extractor</h1>
    <p>NER-powered · Rule-Based Phase 1 · Extracts physio leads from doctor discharge notes</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### 🎯 Confidence Threshold")
    conf_threshold = st.slider("Minimum confidence to flag as Physio lead",
                               min_value=0.50, max_value=1.00, value=0.75, step=0.05,
                               format="%.0%%")

    st.markdown("### 📋 Keyword Dictionary")
    st.markdown(f"**{len(PHYSIO_KEYWORDS)} keywords** loaded")
    with st.expander("View all keywords"):
        for kw in sorted(PHYSIO_KEYWORDS):
            st.markdown(f"• {kw}")

    st.markdown("### ➕ Add Custom Keywords")
    custom_kws = st.text_area("One keyword per line:",
                              placeholder="BALANCE BOARD\nPROPRIOCEPTION\n...", height=100)

    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_details  = st.checkbox("Show match details", value=True)
    show_no_advice = st.checkbox("Show 'No Advice' rows too", value=True)

# ── Custom keyword injection ──────────────────────────────────────────────────
if custom_kws.strip():
    from physio_ner import PHYSIO_KEYWORDS as PKW, matcher, nlp, patterns
    new_kws = [k.strip().upper() for k in custom_kws.strip().split("\n") if k.strip()]
    new_patterns = [nlp.make_doc(k.lower()) for k in new_kws]
    matcher.add("PHYSIO", new_patterns)
    PKW.extend(new_kws)

# ── Helpers ───────────────────────────────────────────────────────────────────
def conf_color(conf: float) -> str:
    if conf >= 0.90: return "#16a34a"
    if conf >= 0.75: return "#f59e0b"
    return "#94a3b8"

def render_result_card(pid: str, result: dict, idx: int):
    label    = result["label"]
    conf     = result["confidence"]
    keywords = result["keywords"]

    if label == "No Advice" and not show_no_advice:
        return

    css_class   = "result-physio" if label == "Physiotherapy" else "result-none"
    label_class = "label-physio"  if label == "Physiotherapy" else "label-none"
    color       = conf_color(conf)
    bar_w       = int(conf * 100)

    kw_html = "".join(f'<span class="kw-chip">{kw}</span>' for kw in keywords) if keywords else "<em style='color:#94a3b8'>—</em>"

    st.markdown(f"""
    <div class="{css_class}">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="font-weight:600;font-size:1rem">{pid}</span>&nbsp;&nbsp;
                <span class="{label_class}">{label}</span>
            </div>
            <div style="text-align:right;font-size:0.82rem;color:#374151">
                Confidence: <strong style="color:{color}">{conf:.0%}</strong>
            </div>
        </div>
        <div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}%;background:{color}"></div></div>
        <div style="margin-top:8px;font-size:0.82rem;color:#374151"><strong>Keywords:</strong><br>{kw_html}</div>
    </div>
    """, unsafe_allow_html=True)

def build_export_df(results_list: list[tuple]) -> pd.DataFrame:
    rows = []
    for pid, res in results_list:
        rows.append({
            "Patient ID":     pid,
            "Label":          res["label"],
            "Confidence":     res["confidence"],
            "Keywords Found": " | ".join(res["keywords"]),
            "Keyword Count":  len(res["keywords"]),
        })
    return pd.DataFrame(rows)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_single, tab_bulk, tab_upload = st.tabs([
    "💬  Single Text", "📋  Bulk Paste", "📂  Upload File"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — SINGLE TEXT ENTRY
# ════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("### Paste a single doctor's discharge note")

    col1, col2 = st.columns([3, 1])
    with col1:
        single_text = st.text_area(
            "Doctor's Advice / Discharge Note",
            placeholder="|PHYSIOTHERAPY FOR BACK AND NECK-TO CONTINUE\n|REVIEW AFTER 1 WEEK WITH DR AJAY HEGDE\n|CHEST PHYSIOTHERAPY: POSTURAL DRAINAGE",
            height=160,
            label_visibility="collapsed"
        )
        patient_id_single = st.text_input("Patient ID (optional):", value="PT001")

    with col2:
        st.markdown("""
        <div class="card" style="height:100%">
            <strong>💡 Tips</strong><br><br>
            • Paste raw text as-is<br>
            • Pipe <code>|</code> delimiters OK<br>
            • Mixed case OK<br>
            • Typos handled ✓
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔍 Extract Physio Lead", type="primary", use_container_width=True):
        if not single_text.strip():
            st.warning("Please enter some text.")
        else:
            result = extract_physio_leads(single_text)
            if result["confidence"] < conf_threshold and result["label"] == "Physiotherapy":
                result["label"] = "No Advice"

            st.markdown("---")
            st.markdown("### Result")
            render_result_card(patient_id_single or "Patient", result, 0)

            if show_details and result["details"]:
                with st.expander("🔬 Match Details (for debugging)"):
                    det_df = pd.DataFrame(result["details"])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)

            with st.expander("📝 Preprocessed Text"):
                st.code(result["clean_text"], language=None)


# ════════════════════════════════════════════════════════════
# TAB 2 — BULK PASTE (multiple patients)
# ════════════════════════════════════════════════════════════
with tab_bulk:
    st.markdown("### Paste multiple records")
    st.markdown("Format: one record per line as `PatientID:::Text` or just paste text lines (auto-numbered)")

    bulk_text = st.text_area(
        "Bulk input",
        placeholder="PT001:::PHYSIOTHERAPY AT HOME, REVIEW AFTER 1 WEEK\nPT002:::CHEST PHYSIOTHERAPY POSTURAL DRAINAGE\nPT003:::REVIEW WITH CARDIOLOGIST, TAB ASPIRIN 75MG",
        height=250,
        label_visibility="collapsed"
    )

    if st.button("🔍 Extract Leads from All Records", type="primary", use_container_width=True):
        if not bulk_text.strip():
            st.warning("Please paste some records.")
        else:
            lines = [l.strip() for l in bulk_text.strip().split("\n") if l.strip()]
            records = []
            for i, line in enumerate(lines):
                if ":::" in line:
                    pid, text = line.split(":::", 1)
                else:
                    pid, text = f"PT{i+1:03d}", line
                records.append({"patient_id": pid.strip(), "text": text.strip()})

            results_list = []
            prog = st.progress(0, text="Processing…")
            for j, rec in enumerate(records):
                res = extract_physio_leads(rec["text"])
                if res["confidence"] < conf_threshold and res["label"] == "Physiotherapy":
                    res["label"] = "No Advice"
                results_list.append((rec["patient_id"], res))
                prog.progress((j+1)/len(records), text=f"Processing {j+1}/{len(records)}…")
            prog.progress(1.0, text="✅ Done!")

            # Stats
            total   = len(results_list)
            physio  = sum(1 for _, r in results_list if r["label"] == "Physiotherapy")
            no_adv  = total - physio

            st.markdown("### 📊 Summary")
            s1, s2, s3 = st.columns(3)
            for col, num, label in [(s1,total,"Total Records"),(s2,physio,"Physio Leads"),(s3,no_adv,"No Advice")]:
                col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

            st.markdown("### 🧑‍⚕️ Results")
            for pid, res in results_list:
                render_result_card(pid, res, 0)

            # Export
            export_df = build_export_df(results_list)
            st.markdown("### 💾 Export")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ CSV", export_df.to_csv(index=False).encode(),
                    f"physio_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
            with c2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    export_df.to_excel(w, index=False, sheet_name="Physio Leads")
                st.download_button("⬇️ Excel", buf.getvalue(),
                    f"physio_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — FILE UPLOAD
# ════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload Excel or CSV file")

    col_up, col_info = st.columns([2,1])
    with col_up:
        uploaded_file = st.file_uploader("Upload file", type=["xlsx","xls","csv"],
                                         label_visibility="collapsed")
    with col_info:
        st.markdown("""
        <div class="card">
            <strong>📋 Expected Format</strong><br><br>
            Needs at least:<br>
            • <code>Patient ID</code> column<br>
            • <code>Discharge Summary</code> / <code>Text</code> column<br><br>
            <span style="font-size:0.82rem;color:#6b7280">Auto-detected column names</span>
        </div>
        """, unsafe_allow_html=True)

    if not uploaded_file:
        sample = pd.DataFrame({
            "Patient ID": ["PT001","PT002","PT003"],
            "Discharge Summary": [
                "PHYSIOTHERAPY AT HOME | REVIEW AFTER 1 WEEK",
                "CHEST PHYSIOTHERAPY: POSTURAL DRAINAGE | INCENTIVE SPIROMETRY",
                "TAB PARACETAMOL 500MG BD | REVIEW WITH CARDIOLOGIST AFTER 2 WEEKS"
            ]
        })
        st.download_button("📥 Download Sample Template",
                           sample.to_csv(index=False).encode(),
                           "sample_physio_template.csv", "text/csv")
    else:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.success(f"✅ Loaded **{len(df)} rows** from `{uploaded_file.name}`")

        # Column detection
        id_cands  = ["patient id","patient_id","patientid","id","mrn","pid","patient number"]
        txt_cands = ["discharge summary","summary","text","notes","discharge_summary","clinical notes","advice","report"]
        id_col = txt_col = None
        for col in df.columns:
            cl = col.strip().lower()
            if cl in id_cands  and id_col  is None: id_col  = col
            if cl in txt_cands and txt_col is None: txt_col = col

        c1, c2 = st.columns(2)
        with c1:
            id_col  = st.selectbox("Patient ID column:", df.columns.tolist(),
                                   index=df.columns.tolist().index(id_col) if id_col else 0)
        with c2:
            txt_col = st.selectbox("Discharge Summary column:", df.columns.tolist(),
                                   index=df.columns.tolist().index(txt_col) if txt_col else min(1,len(df.columns)-1))

        with st.expander("👁️ Preview"):
            st.dataframe(df[[id_col, txt_col]].head(5), use_container_width=True)

        if st.button("🔍 Extract Physio Leads from File", type="primary", use_container_width=True):
            results_list = []
            prog = st.progress(0, text="Starting…")
            total = len(df)
            for i, row in df.iterrows():
                pid  = str(row[id_col])
                text = str(row[txt_col])
                res  = extract_physio_leads(text)
                if res["confidence"] < conf_threshold and res["label"] == "Physiotherapy":
                    res["label"] = "No Advice"
                results_list.append((pid, res))
                prog.progress((i+1)/total, text=f"Processing {i+1}/{total}…")
            prog.progress(1.0, text="✅ Done!")

            physio = sum(1 for _, r in results_list if r["label"] == "Physiotherapy")
            no_adv = total - physio

            st.markdown("### 📊 Summary")
            s1, s2, s3 = st.columns(3)
            for col, num, lbl in [(s1,total,"Total Records"),(s2,physio,"Physio Leads"),(s3,no_adv,"No Advice")]:
                col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("### 🧑‍⚕️ Results")
            for pid, res in results_list:
                render_result_card(pid, res, 0)

            export_df = build_export_df(results_list)
            st.markdown("### 💾 Export")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ CSV", export_df.to_csv(index=False).encode(),
                    f"physio_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
            with c2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    export_df.to_excel(w, index=False, sheet_name="Physio Leads")
                st.download_button("⬇️ Excel", buf.getvalue(),
                    f"physio_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
