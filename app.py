import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="School Dashboard (Base) – Prototype", layout="wide")

DATA_DIR = "data"
LOGO_PATH = "images/nbme_logo.svg"



# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    report_runs = pd.read_csv(os.path.join(DATA_DIR, "report_runs.csv"), parse_dates=["order_date", "test_date"])
    examinees = pd.read_csv(os.path.join(DATA_DIR, "examinees.csv"))
    product_catalog = pd.read_csv(os.path.join(DATA_DIR, "product_catalog.csv"))
    entitlements = pd.read_csv(os.path.join(DATA_DIR, "school_entitlements.csv"))
    return report_runs, examinees, product_catalog, entitlements

report_runs, examinees, product_catalog, entitlements = load_data()

@st.cache_data(show_spinner=False)
def load_scores_for_school(school_id):
    path = os.path.join(DATA_DIR, f"scores_long_{school_id}.csv.gz")
    return pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["test_date"],
        usecols=[
            "school_id", "order_id", "test_date",
            "examinee_id", "examinee_name", "ms_year",
            "metric_label", "metric_group", "value"
        ],
        dtype={
            "school_id": "int64",
            "order_id": "string",
            "examinee_id": "string",
            "examinee_name": "string",
            "ms_year": "category",
            "metric_label": "string",
            "metric_group": "category",
            "value": "float32"
        }
    )



# ----- School naming (UI-only, stable mapping) -----
school_ids = sorted(report_runs["school_id"].unique().tolist())

school_name_map = {
    school_ids[0]: "Redwood University School of Medicine",
    school_ids[1]: "Lakeshore College of Medicine",
    school_ids[2]: "Wharton Street College of Medicine",
    school_ids[3]: "Summit Ridge School of Medicine",
    school_ids[4]: "Pine Valley University College of Medicine",
}

def school_label(sid: str) -> str:
    return f"{school_name_map.get(sid, sid)} ({sid})"


# ----- Session state (Landing + routing) -----
if "school_id" not in st.session_state:
    st.session_state.school_id = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Exam Administrations"
if "landing_exam_filter" not in st.session_state:
    st.session_state.landing_exam_filter = None
if "selected_student_id" not in st.session_state:
    st.session_state.selected_student_id = None
if "selected_order_id" not in st.session_state:
    st.session_state.selected_order_id = None


def _contains(series: pd.Series, query: str) -> pd.Series:
    query = (query or "").strip().lower()
    if not query:
        return pd.Series([False] * len(series), index=series.index)
    return series.fillna("").astype(str).str.lower().str.contains(query, na=False)


# ----- Landing page: choose school first -----
if st.session_state.school_id is None:

    # NBME logo (landing page)
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
        st.markdown(" ")


    st.title("School Dashboard (Base Tier)")
    st.caption("Prototype environment • Demonstration only")
    st.caption("Data shown is synthetic and for demonstration purposes only.")
    st.subheader("Select a school to begin")


    chosen_school = st.radio(
        "Schools",
        options=school_ids,
        format_func=school_label,
        key="landing_school_radio"
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Enter dashboard"):
            st.session_state.school_id = chosen_school
            st.session_state.nav_page = "Exam Administrations"
            st.rerun()

    with c2:
        st.caption("Demo mode: multi-tenant school selection")

    st.stop()




# -------------------------
# Helpers
# -------------------------
def filter_last_24_months(df: pd.DataFrame, date_col: str = "test_date") -> pd.DataFrame:
    max_dt = df[date_col].max()
    if pd.isna(max_dt):
        return df
    cutoff = max_dt - pd.DateOffset(months=24)
    return df[df[date_col] >= cutoff].copy()

def safe_contains(series: pd.Series, q: str) -> pd.Series:
    q = (q or "").strip().lower()
    if not q:
        return pd.Series([True] * len(series), index=series.index)
    return series.fillna("").astype(str).str.lower().str.contains(q, na=False)

def sort_by_last_name(df: pd.DataFrame, name_col: str = "examinee_name") -> pd.DataFrame:
    """
    Sort names alphabetically by last name, then first name.
    Assumes 'First Last' or 'First Middle Last' format.
    """
    df = df.copy()
    parts = df[name_col].astype(str).str.strip().str.split()
    df["_last_name"] = parts.str[-1].str.lower()
    df["_first_name"] = parts.str[0].str.lower()
    df = df.sort_values(by=["_last_name", "_first_name"])
    return df.drop(columns=["_last_name", "_first_name"])


def get_order_context(rr_24: pd.DataFrame, order_id: str) -> dict:
    row = rr_24[rr_24["order_id"] == order_id].head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "order_id": r["order_id"],
        "test_date": r["test_date"],
        "order_date": r["order_date"],
        "exam_name": r["exam_name"],
        "product_family": r["product_family"],
        "product_subfamily": r["product_subfamily"],
        "product_key": r["product_key"],
        "schema_key": r["schema_key"],
        "schema_type": r["schema_type"],
        "n_examinees": int(r["n_examinees"]),
        "school_id": r["school_id"],
    }

def get_order_scores_long(sl_24: pd.DataFrame, order_id: str) -> pd.DataFrame:
    return sl_24[sl_24["order_id"] == order_id].copy()

def choose_total_metric_label(df_order: pd.DataFrame) -> str | None:
    """
    Return the best 'total' metric label from long-form data.
    Tries:
      1) metric_group == TOTAL
      2) label starts with Total
      3) label contains Total (e.g., TotalScaled, TotalPercentCorrect)
    """
    if df_order.empty:
        return None

    # 1) Explicit TOTAL group
    if "metric_group" in df_order.columns:
        totals = df_order[df_order["metric_group"].astype(str).str.upper() == "TOTAL"]
        if not totals.empty:
            # Prefer something that looks like a total score
            # If multiple, take the first consistent
            return str(totals["metric_label"].iloc[0])

    labels = df_order["metric_label"].dropna().astype(str)

    # 2) Startswith "Total"
    starts_total = labels[labels.str.match(r"^Total", case=False)]
    if not starts_total.empty:
        # Prefer scaled if present
        scaled = starts_total[starts_total.str.contains("Scaled", case=False)]
        if not scaled.empty:
            return str(scaled.iloc[0])
        return str(starts_total.iloc[0])

    # 3) Contains "Total" anywhere (CAS often uses TotalScaled / TotalPercentCorrect)
    contains_total = labels[labels.str.contains("Total", case=False)]
    if not contains_total.empty:
        scaled = contains_total[contains_total.str.contains("Scaled", case=False)]
        if not scaled.empty:
            return str(scaled.iloc[0])
        return str(contains_total.iloc[0])

    return None


def order_scores_wide(df_order: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["examinee_id", "examinee_name", "ms_year"]
    keep = df_order[idx_cols + ["metric_label", "value"]].copy()
    wide = keep.pivot_table(index=idx_cols, columns="metric_label", values="value", aggfunc="mean").reset_index()
    wide.columns = [c if isinstance(c, str) else str(c) for c in wide.columns]
    return wide

import re

def prettify_metric_column(col: str) -> str:
    # Keep identity columns untouched
    if col in {"examinee_id", "examinee_name", "ms_year"}:
        return col

    s = str(col)

    # Add space before PercentCorrect / Scaled
    s = re.sub(r"PercentCorrect$", " % Correct", s)
    s = re.sub(r"Scaled$", " Scaled", s)

    # Add space between lower->Upper transitions (e.g., systemPercent -> system Percent)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

    # Normalize slashes spacing
    s = s.replace("/", " / ")

    # Clean up double spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Title case first word only lightly (avoid wrecking acronyms)
    # You can comment this out if you prefer original casing
    s = s.title()


    return s


def clean_roster_table(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()

    # Drop export index columns
    drop_cols = [c for c in wide.columns if str(c).startswith("Unnamed")]

    # Drop duplicated identity fields that sometimes appear as "metrics" (common in CAS-style exports)
    id_like = {
        "ExamineeId", "FirstName", "LastName", "MidName",
        "School Id", "Exam Name",
        "ActualTestDate", "StartDate", "EndDate"
    }
    drop_cols += [c for c in wide.columns if c in id_like]

    wide = wide.drop(columns=list(set(drop_cols)), errors="ignore")

    # Reorder: identity columns first
    first = [c for c in ["examinee_id", "examinee_name", "ms_year"] if c in wide.columns]
    others = [c for c in wide.columns if c not in first]
    wide = wide[first + others]

    # Round numeric columns for readability
    for c in others:
        if pd.api.types.is_numeric_dtype(wide[c]):
            wide[c] = wide[c].round(1)

    # Prettify metric column names for display
    wide = wide.rename(columns={c: prettify_metric_column(c) for c in wide.columns})


    return wide

def is_valid_subscore_label(label: str) -> bool:
    """
    Exclude identity, metadata, and total fields from subscore previews.
    """
    bad_patterns = [
        "Name", "Id", "Date", "Mid", "First", "Last",
        "School", "Exam", "Start", "End",
        "Total"
    ]
    return not any(p.lower() in str(label).lower() for p in bad_patterns)



def metric_groups_for_order(df_order: pd.DataFrame) -> list[str]:
    if df_order.empty or "metric_group" not in df_order.columns:
        return ["All metrics"]
    groups = df_order["metric_group"].fillna("OTHER").astype(str).unique().tolist()
    groups = sorted(groups, key=lambda x: (x != "TOTAL", x))
    return ["All metrics"] + groups



def filter_wide_by_threshold(wide: pd.DataFrame, metric_col: str | None, op: str, threshold: float | None) -> pd.DataFrame:
    if wide.empty or not metric_col or metric_col not in wide.columns or threshold is None:
        return wide
    s = wide[metric_col]
    if op == "<":
        return wide[s < threshold]
    if op == "<=":
        return wide[s <= threshold]
    if op == ">":
        return wide[s > threshold]
    if op == ">=":
        return wide[s >= threshold]
    return wide

def render_hist(series: pd.Series, title: str):
    fig = plt.figure()
    plt.hist(series.dropna().values, bins=30)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(title)
    st.pyplot(fig)


# ----- NBME logo (top-left) -----
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    st.sidebar.markdown("---")

# -------------------------
# Base Tier shell / state
# -------------------------
school_id = st.session_state.school_id

st.sidebar.markdown(f"**School:** {school_label(school_id)}")
if st.sidebar.button("Switch school"):
    st.session_state.school_id = None
    st.session_state.nav_page = "Exam Administrations"
    st.session_state.landing_exam_filter = None
    st.session_state.selected_student_id = None
    st.session_state.selected_order_id = None
    st.rerun()

st.title("School Dashboard (Base Tier)")

st.markdown(
    """
    <div style="
        display:inline-block;
        padding:2px 8px;
        border-radius:999px;
        font-size:12px;
        color:#4b5563;
        background:#f3f4f6;
        border:1px solid #e5e7eb;
        margin-top:-6px;
        ">
        Prototype
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Base Tier: in-product score visibility (no trends, no benchmarks, no automation).",
    help=(
        "Included in Base:\n"
        "• Score Reports across Subject Exams, CAS, and NSAS\n"
        "• In-product tabular 'Download Scores' view\n"
        "• Per-exam score distribution (histogram)\n"
        "• Student search with 24-month exam history\n\n"
        "Not included in Base:\n"
        "• Multi-year trends or cohort comparisons\n"
        "• National norms or benchmarking\n"
        "• Predictive analytics\n"
        "• Bulk workflows, exports, or integrations"
    )
)

st.caption("Data shown is synthetic.")
st.caption(f"Current school: {school_label(school_id)}")



# 24-month constraint (Base)
rr_school = filter_last_24_months(
    report_runs[report_runs["school_id"] == school_id].copy(),
    "test_date"
)

scores_long = load_scores_for_school(school_id)
sl_school = filter_last_24_months(scores_long.copy(), "test_date")

ex_school = examinees[examinees["school_id"] == school_id].copy()


# Navigation (respect routed destination)
nav = st.sidebar.radio(
    "Navigate",
    ["Exam Administrations", "Students"],
    key="nav_page"
)


# -------------------------
# SCORE REPORTS JOURNEY
# -------------------------
def page_score_reports():
    st.subheader("Exam Administrations (last 24 months)")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 2.0])

    fams = ["All"] + sorted(rr_school["product_family"].unique().tolist())
    with c1:
        fam = st.selectbox("Product family", fams)

    df_f = rr_school.copy()

    # If landing page routed via exam name, apply it once
    if st.session_state.landing_exam_filter:
        df_f = df_f[df_f["exam_name"] == st.session_state.landing_exam_filter]
        st.session_state.landing_exam_filter = None

    if fam != "All":
        df_f = df_f[df_f["product_family"] == fam]

    subfams = ["All"] + sorted(df_f["product_subfamily"].unique().tolist())
    with c2:
        subfam = st.selectbox("Subfamily", subfams)

    if subfam != "All":
        df_f = df_f[df_f["product_subfamily"] == subfam]

    exams = ["All"] + sorted(df_f["exam_name"].unique().tolist())
    with c3:
        exam = st.selectbox("Exam", exams)

    if exam != "All":
        df_f = df_f[df_f["exam_name"] == exam]

    with c4:
        q = st.text_input("Search (order id / exam)", value="", placeholder="e.g., OB- or Medicine")

    if q.strip():
        mask = safe_contains(df_f["order_id"], q) | safe_contains(df_f["exam_name"], q)
        df_f = df_f[mask]

    df_f = df_f.sort_values("test_date", ascending=False)

    st.caption(f"{len(df_f):,} score reports found.")
    if df_f.empty:
        st.info("No score reports match your filters. Try clearing filters.")
        return

    st.dataframe(
        df_f.rename(columns={
            "order_id": "Order ID",
            "test_date": "Test Date",
            "exam_name": "Exam",
            "product_family": "Family",
            "product_subfamily": "Subfamily",
            "n_examinees": "# Examinees",
        })[["Order ID", "Test Date", "Exam", "Family", "Subfamily", "# Examinees"]],
        use_container_width=True,
        height=320
    )

    # Open a report — respect landing-page routing
    order_list = df_f["order_id"].tolist()
    default_order = st.session_state.selected_order_id
    default_index = 0
    if default_order and default_order in order_list:
        default_index = order_list.index(default_order)

    chosen_order = st.selectbox("View exam administration", order_list, index=default_index)
    st.session_state.selected_order_id = chosen_order

    render_exam_admin_detail(chosen_order)


# -------------------------
# STUDENT SEARCH JOURNEY
# -------------------------
def page_student_search():
    st.subheader("Student Search → 24-Month Exam History")

    q = st.text_input(
        "Search student name",
        value="",
        placeholder="Type last name, first name, or partial (e.g., Wong, Mart, al-)"
    )

    if q.strip():
        matches = ex_school[safe_contains(ex_school["examinee_name"], q)].copy()
    else:
        matches = ex_school.sample(min(50, len(ex_school)), random_state=1).copy()

    # Sort alphabetically by last name for usability
    matches = sort_by_last_name(matches)


    st.caption(f"{len(matches):,} matching students" if q.strip() else "Showing a sample of students. Type to search.")
    if matches.empty:
        st.info("No students match that search.")
        return

    forced_id = st.session_state.selected_student_id

    if forced_id:
        forced = ex_school[ex_school["examinee_id"] == forced_id]
        if not forced.empty:
            sel_name = forced["examinee_name"].iloc[0]
        else:
            sel_name = st.selectbox("Select a student", matches["examinee_name"].tolist())
    else:
        sel_name = st.selectbox("Select a student", matches["examinee_name"].tolist())

    # IMPORTANT: look up the selected student in the full school roster, not just matches
    sel = ex_school[ex_school["examinee_name"] == sel_name].head(1)
    if sel.empty:
        st.error("Could not find selected student in this school roster.")
        return

    examinee_id = sel["examinee_id"].iloc[0]
    st.session_state.selected_student_id = examinee_id

    a, b, c = st.columns([1.2, 1.2, 1.6])
    with a:
        st.metric("Student", sel_name)
    with b:
        st.metric("Examinee ID", examinee_id)
    with c:
        st.metric("MS Year", sel["ms_year"].iloc[0] if "ms_year" in sel.columns else "—")

    orders = (
        sl_school[sl_school["examinee_id"] == examinee_id][["order_id", "test_date"]]
        .drop_duplicates()
        .merge(rr_school[["order_id", "exam_name", "product_family", "product_subfamily"]], on="order_id", how="left")
        .sort_values("test_date", ascending=False)
    )

    st.markdown("### Exam history (last 24 months)")
    if orders.empty:
        st.info("No exams found for this student in the last 24 months.")
        return

    st.dataframe(
        orders.rename(columns={
            "test_date": "Test Date",
            "exam_name": "Exam",
            "product_family": "Family",
            "product_subfamily": "Subfamily",
            "order_id": "Order ID"
        }),
        use_container_width=True,
        height=280
    )

    chosen_order = st.selectbox("Open an exam from this history", orders["order_id"].tolist())
    st.session_state.selected_order_id = chosen_order
    render_exam_admin_detail(chosen_order, highlight_examinee_id=examinee_id)


# -------------------------
# EXAM ADMIN DETAIL (Base Tier core)
# -------------------------
def render_exam_admin_detail(order_id: str, highlight_examinee_id: str | None = None):
    ctx = get_order_context(rr_school, order_id)
    if not ctx:
        st.error("Could not load this score report.")
        return

    st.markdown("---")
    st.markdown("## Selected score report")

    # Context line (explicitly ties detail panel to the selection above)
    test_date_str = ctx["test_date"].date().strftime("%B %d, %Y") if pd.notna(ctx["test_date"]) else "—"
    st.markdown(
        f"**{ctx['exam_name']}** · {test_date_str} · Order `{ctx['order_id']}` · {ctx['n_examinees']} examinees"
    )
    st.caption("This section shows detailed results for the selected exam administration.")



    h1, h2, h3, h4 = st.columns([2.5, 1.2, 1.2, 1.2])
    with h1:
        st.markdown(f"**{ctx['product_family']} → {ctx['product_subfamily']}**")
    with h2:
        st.metric("Test date", ctx["test_date"].date().isoformat() if pd.notna(ctx["test_date"]) else "—")
    with h3:
        st.metric("Order ID", ctx["order_id"])
    with h4:
        st.metric("# Examinees", ctx["n_examinees"])

    df_order = get_order_scores_long(sl_school, order_id)
    if df_order.empty:
        st.info("No scores found for this administration.")
        return

    total_label = choose_total_metric_label(df_order)
    wide = order_scores_wide(df_order)
    wide = clean_roster_table(wide)

    # Build mapping from raw metric_label -> prettified column name
    raw_to_pretty = {
        raw: prettify_metric_column(raw)
        for raw in df_order["metric_label"].dropna().unique()
    }


    # Fallback: if we couldn't detect total from long labels, detect from wide columns
    if not total_label or total_label not in wide.columns:
        total_candidates = [c for c in wide.columns if "Total" in str(c)]
        if total_candidates:
            # Prefer Total Scaled over Total % Correct
            scaled = [c for c in total_candidates if "Scaled" in str(c)]
            pct = [c for c in total_candidates if "% Correct" in str(c)]
            if scaled:
                total_label = scaled[0]
            elif pct:
                total_label = pct[0]
            else:
                total_label = total_candidates[0]



    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 2.1])
    with c1:
        groups = metric_groups_for_order(df_order)
        label_map = {
            "CAS_DOMAIN": "CAS Domains",
            "SYSTEM_OR_DOMAIN": "Systems / Domains",
            "PHYSICIAN_TASK": "Physician Tasks",
            "PATIENT_CARE_TASK": "Patient Care Tasks",
            "TOTAL": "Total"
        }

        selected_group = st.selectbox(
            "Metric group",
            groups,
            index=0,
            format_func=lambda g: label_map.get(g, g)
        )

    with c2:
        roster_search = st.text_input("Search roster", value="", placeholder="Search student name")
    with c3:
        cols = [c for c in wide.columns if c not in ["examinee_id", "examinee_name", "ms_year"]]
        if total_label and total_label in wide.columns:
            threshold_metric = st.selectbox("Threshold metric", [total_label] + cols[:10])
        else:
            threshold_metric = st.selectbox("Threshold metric", cols[:10] if cols else ["(no metric)"])
    
    with c4:
        tcol1, tcol2, tcol3 = st.columns([0.7, 1.0, 1.4])
        with tcol1:
            op = st.selectbox("Op", ["<", "<=", ">", ">="], index=0)
        with tcol2:
            thr = st.number_input("Threshold", value=55.0, step=1.0)
        with tcol3:
            apply_threshold = st.checkbox("Apply threshold", value=False)
            st.caption("Tip: Use this to identify at-risk students quickly.")


    left, right = st.columns([1.1, 1.9])

    with left:
        st.markdown("### Distribution (Total)")
        if total_label and total_label in wide.columns:
            render_hist(wide[total_label], f"Total: {total_label}")
        else:
            st.info("TOTAL metric not found for this administration.")

        st.markdown("### Subscore preview")
        if selected_group != "All metrics" and "metric_group" in df_order.columns:
            grp = df_order[
                (df_order["metric_group"].astype(str) == selected_group) &
                (df_order["metric_label"].apply(is_valid_subscore_label))
            ]
            if grp.empty:
                st.caption("No metrics in this group for this exam.")
            else:
                summary = grp.groupby("metric_label")["value"].mean().sort_values(ascending=False).head(6).reset_index()
                summary["mean"] = summary["value"].round(1)
                summary["metric_label"] = summary["metric_label"].apply(prettify_metric_column)
                st.dataframe(summary[["metric_label", "mean"]], use_container_width=True, height=240)

        else:
            st.caption("Select a metric group to preview subscores.")

    with right:
        st.markdown("### Roster (Download Scores)")
        roster = wide.copy()

        if selected_group != "All metrics" and "metric_group" in df_order.columns:
            allowed_raw = df_order[
                (df_order["metric_group"].astype(str) == selected_group) &
                (df_order["metric_label"].apply(is_valid_subscore_label))
            ]["metric_label"].unique().tolist()

            # Convert raw labels to prettified roster column names
            allowed_pretty = [
                raw_to_pretty[r] for r in allowed_raw if r in raw_to_pretty
            ]

            keep_cols = ["examinee_id", "examinee_name", "ms_year"] + [
                c for c in roster.columns if c in allowed_pretty
            ]

            if len(keep_cols) <= 3 and total_label in roster.columns:
                keep_cols += [total_label]
            roster = roster[keep_cols]

        if roster_search.strip():
            roster = roster[safe_contains(roster["examinee_name"], roster_search)]

        if apply_threshold:
            roster = filter_wide_by_threshold(roster, threshold_metric, op, thr)


        if highlight_examinee_id:
            st.caption("Highlighted student from search will appear at top when present in this exam.")
            if highlight_examinee_id in roster["examinee_id"].values:
                top = roster[roster["examinee_id"] == highlight_examinee_id]
                rest = roster[roster["examinee_id"] != highlight_examinee_id]
                roster = pd.concat([top, rest], ignore_index=True)

        MAX_ROWS_DEFAULT = 500
        show_all = st.checkbox("Show full roster", value=False)

        display_roster = roster if show_all else roster.head(MAX_ROWS_DEFAULT)

        st.caption(
            f"{len(roster):,} students matched."
            + ("" if show_all else f" Showing first {MAX_ROWS_DEFAULT:,}.")
        )

        st.dataframe(display_roster, width="stretch", height=420)


        st.download_button(
            "Export roster as CSV",
            data=roster.to_csv(index=False).encode("utf-8"),
            file_name=f"download_scores_{order_id}.csv",
            mime="text/csv"
        )


# -------------------------
# Footer (Last updated)
# -------------------------
st.markdown("---")
st.caption(f"Last updated: {date.today().isoformat()}")


# -------------------------
# Render page
# -------------------------
if nav == "Exam Administrations":
    page_score_reports()
else:
    page_student_search()



