"""
physio_ner.py
=============
Rule-Based NER Pipeline for Physiotherapy Lead Extraction
Uses spaCy PhraseMatcher (exact) + RapidFuzz (fuzzy) on doctor advice notes.

Author  : Built for Discharge Summary Lead Extractor
Version : 1.0  (Rule-Based — Phase 1)
"""

import re
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz, process
import pandas as pd
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PHYSIOTHERAPY KEYWORD DICTIONARY
#     Source: provided by domain expert (doctor notes vocabulary)
# ──────────────────────────────────────────────────────────────────────────────

PHYSIO_KEYWORDS = [
    # Exact terms from dictionary
    "BED SIDE MOBILISATION",
    "BEDSIDE MOBILISATION",
    "FULL WEIGHT BEARING WALKING WITH WALKER",
    "ARM POUCH MOBILIZATION",
    "ANTI DVT EXERCISES",
    "STATIC QUADS AND HAMSTRING EXERCISES",
    "SHOULDER PENDULUM EXERCISES",
    "SHOULDER PENDULUM EXCERCISES",         # common misspelling
    "SQE",
    "ISOMETRIC QUADRICEPS EXERCISES",
    "STATIC QUADRICEPS EXERCISES",
    "FULL WEIGHT BEARING AMBULATION",
    "ACTIVE AND PASSIVE SHOULDER ROM EXERCISES",
    "PENDULUM EXERCISES",
    "ROM",
    "RANGE OF MOTION",
    "ELECTROTHERAPY",
    "MAGNETOTHERAPY",
    "HEAD END ELEVATION",
    "FOOT END ELEVATION",
    "SLR",
    "STRAIGHT LEG RAISE",
    "STRAIGHT LEG RISE",
    "STATIC QUADS EXERCISES",
    "STATIC QUADS",
    "LEFT LOWER LIMB AND TRUNK MUSCLES",
    "AAROM",
    "ASSISTED STRENGTHENING",
    "STRETCHING",
    "PROM EXERCISES",
    "PROM",
    "PASSIVE RANGE OF MOTION",
    "KNEE BENDING",
    "ACTIVE KNEE BENDING EXERCISES",
    "ACTIVE KNEE BENDING",
    "BEDSIDE ACTIVE AND AA ROM EXERCISES",
    "ACTIVE ASSISTED ROM EXERCISES",
    "AMBULATION FWB WITH WALKER",
    "AMBULATION WITH WALKER",
    "NON WEIGHT BEARING MOBILISATION WITH WALKER",
    "NON-WEIGHT BEARING MOBILISATION",
    "SWALLOWING THERAPY",
    "QUADS HAMS VMO STRETCHING",
    "HAMSTRING STRETCHING",
    "VMO STRETCHING",
    "ANKLE ROM EXERCISES",
    "ANKLE ROM",
    "FULL WEIGHT BEARING MOBILIZATION",
    "FULL-WEIGHT BEARING MOBILIZATION",
    "FWB MOBILIZATION",
    "AMBULATION",
    "FISTULA EXERCISES",
    "RIGHT LOWER LIMB ELEVATION",
    "LOWER LIMB ELEVATION",
    "LIMB ELEVATION",
    "STUMP EXERCISE",
    "STUMP EXERCISES",
    "AMPUTATION STUMP EXERCISE",
    "QUANTITATIVE SENSORY TESTING",
    "QST",
    "ANKLE PUMPS",
    "PHYSIOTHERAPY",
    "PHYSIO",
    "PHYSICAL THERAPY",
    "EXERCISE",
    "EXERCISES",
    "REHABILITATION",
    "REHAB",
    "MOBILISATION",
    "MOBILIZATION",
    "GAIT TRAINING",
    "STRENGTHENING EXERCISES",
    "STRENGTHENING",
    "WALKER",
    "CRUTCHES",
    "WEIGHT BEARING",
    "WEIGHT-BEARING",
    "CHEST PHYSIOTHERAPY",
    "CHEST PHYSIO",
    "POSTURAL DRAINAGE",
    "INCENTIVE SPIROMETRY",
    "BREATHING EXERCISES",
    "DEEP BREATHING",
    "BALANCE TRAINING",
    "NEUROMUSCULAR",
    "OCCUPATIONAL THERAPY",
    "SPEECH THERAPY",
    "TRACTION",
    "ULTRASOUND THERAPY",
    "TENS",
    "IFT",
    "HOT PACK",
    "COLD PACK",
    "CRYOTHERAPY",
    "WRIST EXTENSION EXERCISES",
    "FINGER EXERCISES",
    "HIP EXERCISES",
    "SHOULDER EXERCISES",
    "BACK EXERCISES",
    "NECK EXERCISES",
    "CORE STRENGTHENING",
    "GUIDED PHYSIOTHERAPY",
    "PHYSIOTHERAPY AT HOME",
    "HOME PHYSIOTHERAPY",
    "PHYSIOTHERAPY FOR BACK AND NECK",
    "PHYSIOTHERAPY FOR BACK",
    "PHYSIOTHERAPY FOR NECK",
]

# Medical exclusion terms — these indicate NON-physio medical activities
# If found in the same segment, invalidate the physio match
MEDICAL_EXCLUSIONS = [
    "tooth extraction", "dental extraction", "extraction of tooth",
    "meet nephrologist", "meet cardiologist", "meet neurologist",
    "meet endocrinologist", "meet surgeon", "meet dentist", "meet dental",
    "opd with", "opd after", "review with dr", "follow up with dr",
    "consultation with", "consult with", "appointment with",
]

# Fuzzy matching threshold — tune this to reduce false positives
# 85 = fairly strict, 75 = more lenient (catches more typos)
FUZZY_THRESHOLD = 82

# Confidence bands
CONF_EXACT   = 1.00   # exact / normalized match
CONF_HIGH    = 0.90   # high fuzzy similarity
CONF_MEDIUM  = 0.75   # medium similarity


# ──────────────────────────────────────────────────────────────────────────────
# 2.  TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Normalize raw doctor notes for matching:
    - Strip pipe delimiters used in many EMR exports
    - Collapse extra whitespace
    - Uppercase (our dictionary is uppercase)
    - Remove special unicode artefacts (e.g. â€™ from copy-paste)
    """
    if not isinstance(text, str):
        return ""
    # Remove unicode noise
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Replace pipe delimiters and newlines with spaces
    text = re.sub(r"[|\n\r]+", " ", text)
    # Remove special chars except hyphen and slash (clinically meaningful)
    text = re.sub(r"[^A-Za-z0-9 \-/\.,()]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def tokenize_segments(text: str) -> list[str]:
    """
    Split the note into logical segments on common delimiters.
    Doctors often write advice as bullet-style lines separated by | or newlines.
    We process each segment independently for better precision.
    """
    raw_segs = re.split(r"[|;\n\r]", text)
    return [s.strip() for s in raw_segs if s.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BUILD spaCy PHRASE MATCHER  (exact + normalized)
# ──────────────────────────────────────────────────────────────────────────────

nlp = spacy.blank("en")

# Add sentencizer so we can process spans
nlp.add_pipe("sentencizer")

# Build PhraseMatcher on lowercase attr so matching is case-insensitive
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

patterns = [nlp.make_doc(kw.lower()) for kw in PHYSIO_KEYWORDS]
matcher.add("PHYSIO", patterns)


def exact_match(text: str) -> list[dict]:
    """
    Run spaCy PhraseMatcher on preprocessed text.
    Returns list of {matched_text, start, end, confidence}.
    """
    doc     = nlp(text.lower())
    matches = matcher(doc)
    results = []
    seen    = set()
    for match_id, start, end in matches:
        span = doc[start:end].text.upper()
        if span not in seen:
            seen.add(span)
            results.append({
                "matched_text": span,
                "match_type":   "EXACT",
                "confidence":   CONF_EXACT,
            })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FUZZY MATCHER  (catches typos, abbreviations, partial terms)
# ──────────────────────────────────────────────────────────────────────────────

# Pre-build uppercase keyword list for fuzzy matching
KEYWORDS_UPPER = [kw.upper() for kw in PHYSIO_KEYWORDS]


def fuzzy_match_segment(segment: str) -> list[dict]:
    """
    For a single text segment, slide a window of n-grams and compare
    each against the keyword dictionary using RapidFuzz.
    Multi-word keywords get multi-word windows; single words get single tokens.
    """
    tokens  = segment.split()
    results = []
    seen    = set()

    # Window sizes: 1 to 6 words (covers longest keywords)
    for window in range(1, 7):
        for i in range(len(tokens) - window + 1):
            chunk = " ".join(tokens[i : i + window])
            
            # Skip very short chunks OR single-word chunks under 5 chars
            # This prevents "TRACTION" matching "extraction" partially
            if len(chunk) < 3 or (window == 1 and len(chunk) < 5):
                continue

            best = process.extractOne(
                chunk,
                KEYWORDS_UPPER,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )

            if best:
                kw, score, _ = best
                
                # Additional validation: if match is single-word and significantly shorter
                # than keyword, it's likely a partial match — skip it
                if window == 1 and len(chunk) < len(kw) * 0.7:
                    continue
                    
                if kw not in seen:
                    seen.add(kw)
                    conf = (
                        CONF_EXACT   if score >= 98 else
                        CONF_HIGH    if score >= 90 else
                        CONF_MEDIUM
                    )
                    results.append({
                        "matched_text": chunk,        # what was found in text
                        "keyword":      kw,           # dictionary keyword it matched
                        "match_type":   "FUZZY",
                        "fuzzy_score":  score,
                        "confidence":   conf,
                    })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CONTEXT VALIDATOR
#     Avoid false positives — e.g. "EXERCISE" alone in a non-physio context
# ──────────────────────────────────────────────────────────────────────────────

# Weak standalone words that need supporting context to count
WEAK_TERMS = {"EXERCISE", "EXERCISES", "ROM", "WALKER", "AMBULATION", "STRETCHING", "REHAB"}

# Strong terms — presence alone is sufficient
STRONG_TERMS = {
    "PHYSIOTHERAPY", "PHYSIO", "PHYSICAL THERAPY", "CHEST PHYSIOTHERAPY",
    "MOBILISATION", "MOBILIZATION", "ELECTROTHERAPY", "MAGNETOTHERAPY",
    "GUIDED PHYSIOTHERAPY", "HOME PHYSIOTHERAPY", "PHYSIOTHERAPY AT HOME",
    "GAIT TRAINING", "SLR", "AAROM", "PROM", "SQE", "QST",
    "POSTURAL DRAINAGE", "INCENTIVE SPIROMETRY",
}

def is_valid_physio_hit(matched_kw: str, all_matches: list[dict], segment_text: str = "") -> bool:
    """
    If a match is a weak term, require at least one other physio match nearby.
    Strong terms are always valid.
    Also check for medical exclusion terms in the segment.
    """
    kw = matched_kw.upper()
    
    # Check for medical exclusion patterns in the segment
    if segment_text:
        seg_lower = segment_text.lower()
        for exclusion in MEDICAL_EXCLUSIONS:
            if exclusion in seg_lower:
                # This segment is about a medical appointment/procedure, not physio
                return False
    
    if kw in STRONG_TERMS:
        return True
    if kw in WEAK_TERMS:
        # valid only if there is at least one other match
        return len(all_matches) > 1
    return True   # all other terms are valid by default


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN EXTRACTION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_physio_leads(raw_text: str) -> dict:
    """
    Full pipeline for one patient text.

    Returns
    -------
    {
        "label"      : "Physiotherapy" | "No Advice",
        "confidence" : float  0.0–1.0,
        "keywords"   : list of matched keyword strings,
        "details"    : list of full match dicts (for debugging/training),
        "clean_text" : normalized text used for matching
    }
    """
    if not raw_text or not raw_text.strip():
        return {
            "label":      "No Advice",
            "confidence": 0.0,
            "keywords":   [],
            "details":    [],
            "clean_text": "",
        }

    clean = preprocess(raw_text)

    # ── Step A: Exact matching on full text ──
    exact_hits = exact_match(clean)

    # ── Step B: Fuzzy matching per segment + store segment map for validation ──
    segments      = tokenize_segments(raw_text)  # use original for segments
    fuzzy_hits    = []
    hit_to_seg    = {}  # track which segment each hit came from
    
    for seg in segments:
        seg_clean = preprocess(seg)
        if seg_clean:
            seg_hits = fuzzy_match_segment(seg_clean)
            for hit in seg_hits:
                hit_to_seg[hit.get("keyword", hit["matched_text"])] = seg_clean
            fuzzy_hits.extend(seg_hits)

    # ── Step C: Merge, deduplicate ──
    all_hits = exact_hits + fuzzy_hits

    # Deduplicate by matched keyword (keep highest confidence)
    seen_kw  = {}
    for hit in all_hits:
        kw = hit.get("keyword", hit["matched_text"])
        if kw not in seen_kw or hit["confidence"] > seen_kw[kw]["confidence"]:
            seen_kw[kw] = hit

    deduped = list(seen_kw.values())

    # ── Step D: Context validation ──
    valid_hits = []
    for h in deduped:
        kw = h.get("keyword", h["matched_text"])
        # Get the segment this keyword was found in (if from fuzzy matching)
        segment_text = hit_to_seg.get(kw, clean)
        if is_valid_physio_hit(kw, deduped, segment_text):
            valid_hits.append(h)

    if not valid_hits:
        return {
            "label":      "No Advice",
            "confidence": 0.0,
            "keywords":   [],
            "details":    [],
            "clean_text": clean,
        }

    # ── Step E: Aggregate confidence ──
    # Use max confidence of strongest hit, boosted by number of hits
    max_conf    = max(h["confidence"] for h in valid_hits)
    hit_count   = len(valid_hits)
    # Small boost for multiple corroborating signals, cap at 1.0
    boost       = min(0.05 * (hit_count - 1), 0.10)
    final_conf  = min(round(max_conf + boost, 2), 1.0)

    # Collect clean keyword labels for display
    keyword_labels = sorted(set(
        h.get("keyword", h["matched_text"]) for h in valid_hits
    ))

    return {
        "label":      "Physiotherapy",
        "confidence": final_conf,
        "keywords":   keyword_labels,
        "details":    valid_hits,
        "clean_text": clean,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_batch(records: list[dict], text_col: str = "text", id_col: str = "patient_id") -> pd.DataFrame:
    """
    Process a list of records [{patient_id, text}, ...] or a flat list of strings.

    Returns a DataFrame with columns:
        patient_id | label | confidence | keywords_found | keyword_count | raw_text
    """
    rows = []
    for rec in records:
        if isinstance(rec, str):
            pid, text = f"ROW_{len(rows)+1}", rec
        else:
            pid  = rec.get(id_col, f"ROW_{len(rows)+1}")
            text = rec.get(text_col, "")

        result = extract_physio_leads(text)
        rows.append({
            "patient_id":     pid,
            "label":          result["label"],
            "confidence":     result["confidence"],
            "keywords_found": " | ".join(result["keywords"]),
            "keyword_count":  len(result["keywords"]),
            "raw_text":       str(text)[:300],
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  QUICK SELF-TEST  (run this file directly to validate)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_CASES = [
        {
            "id": "T1 — No Physio (multi OPD reviews)",
            "text": "|REVIEW AFTER 10 DAYS WITH DR AJAY HEGDE IN NEUROSURGERY OPD|REPEAT SERUM ELECTROLYTES AFTER 1 WEEK|REVIEW WITH DR NAVEEN KUMAR IN ORTHOPAEDICS OPD|REVIEW WITH DR HARSHA IN UROLOGY OPD AFTER 1 WEEK FOR TRIAL VOID"
        },
        {
            "id": "T2 — Physio (chest physio + spirometry)",
            "text": "|HOME MONITORING OF BP/HR|MONITOR GRBS||Review after 4 days with CBC, RP-1|||CARDIAC ADVICE:||DISCONTINUE HOLTER ON 20/05/2025 @ 6.20PM|SELF MONITORING OF BP|REVIEW AT CARDIO OPD WITH RFT REPORT AND BP CHART||RESPIRATORY ADVICE:||CHEST PHYSIOTHERAPY: POSTURAL DRAINAGE|INCENTIVE SPIROMETRY"
        },
        {
            "id": "T3 — Physio (home physio)",
            "text": "|OPD FOLLOW UP AFTER 1 WEEK||PHYSIOTHERAPY AT HOME"
        },
        {
            "id": "T4 — Physio (back and neck)",
            "text": "|PHYSIOTHERAPY FOR BACK AND NECK-TO CONTINUE|VIDEO CONSULTATION AFTER 1 MONTH- DR AJAY HEGDE NEUROSURGERY"
        },
        {
            "id": "T5 — Physio (guided physio)",
            "text": "|Review after 2 weeks for 2nd dose of INJ. RITUXIMAB.||NEEDS GUIDED PHYSIOTHERAPY once hip pain subsides |Osteoporosis Evaluation Next visit. |Vaccinations as per plan"
        },
        {
            "id": "T6 — Physio (SLR + ankle pumps + walker)",
            "text": "PATIENT ADVISED SLR EXERCISES, ANKLE PUMPS, FULL WEIGHT BEARING WALKING WITH WALKER UNDER SUPERVISION"
        },
        {
            "id": "T7 — Physio (typo: EXCERCISES)",
            "text": "SHOULDER PENDULUM EXCERCISES TO BE DONE TWICE DAILY"
        },
        {
            "id": "T8 — No Physio (only medications)",
            "text": "TAB PARACETAMOL 500MG BD, INJ CEFTRIAXONE 1GM IV, MONITOR VITALS, DISCHARGE ON ORAL ANTIBIOTICS"
        },
        {
            "id": "T9 — Edge: EXERCISE alone (should need context)",
            "text": "PATIENT ADVISED EXERCISE AND DIET CONTROL FOR DIABETES MANAGEMENT"
        },
        {
            "id": "T10 — Physio (abbreviation AAROM + ROM)",
            "text": "BEDSIDE AAROM EXERCISES, ROM TO BE MAINTAINED, GAIT TRAINING WITH WALKER ONCE STABLE"
        },
    ]

    print("=" * 75)
    print("  PHYSIOTHERAPY NER — RULE-BASED PIPELINE  |  SELF-TEST")
    print("=" * 75)

    records = [{"patient_id": t["id"], "text": t["text"]} for t in TEST_CASES]
    df = process_batch(records)

    for _, row in df.iterrows():
        status = "✅ PHYSIO" if row["label"] == "Physiotherapy" else "⬜ NO ADVICE"
        print(f"\n{status}  [{row['confidence']:.0%}]  {row['patient_id']}")
        if row["keywords_found"]:
            print(f"   Keywords : {row['keywords_found']}")
        print(f"   Text     : {row['raw_text'][:100]}...")

    print("\n" + "=" * 75)
    print(f"  SUMMARY: {(df['label']=='Physiotherapy').sum()} / {len(df)} flagged as Physiotherapy leads")
    print("=" * 75)

    # Save test output
    df.to_csv("/home/claude/test_output.csv", index=False)
    print("\n✅ Test output saved to test_output.csv")
