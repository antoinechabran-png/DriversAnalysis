import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from semopy import Model
import re
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Consumer Driver Analysis Tool", layout="wide")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
    return output.getvalue()


def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', str(name))


def run_rwa(X, y):
    # Get direction from simple correlation first
    directions = X.apply(lambda col: np.sign(np.corrcoef(col, y)[0, 1]))

    corr_matrix = X.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    diagonal_sqrt_evals = np.diag(np.sqrt(eigenvalues))
    delta = eigenvectors @ diagonal_sqrt_evals @ eigenvectors.T
    transformed_X = np.linalg.inv(delta) @ X.T
    model = sm.OLS(y, sm.add_constant(transformed_X.T)).fit()
    raw_weights = (delta**2) @ (model.params.iloc[1:].values**2)
    rescaled_weights = (raw_weights / raw_weights.sum()) * 100

    res = pd.DataFrame({
        'Driver': X.columns,
        'Weight (%)': rescaled_weights,
        'Direction': directions.map({1.0: 'Positive', -1.0: 'Negative', 0.0: 'Neutral'}).values
    }).sort_values(by='Weight (%)', ascending=False)
    return res


def compute_shapley_like(X, y):
    """Pseudo-Shapley importance: |standardized coefficient|, rescaled to sum to 100%."""
    model = sm.OLS(y, sm.add_constant(X)).fit()
    raw_std_coefs = model.params.iloc[1:] * (X.std() / y.std())
    std_coefs_abs = np.abs(raw_std_coefs)
    shap_pct = (std_coefs_abs / std_coefs_abs.sum()) * 100
    shap_pct.index = X.columns
    raw_std_coefs.index = X.columns
    return shap_pct, raw_std_coefs


def bootstrap_weights(X, y, method_func, n_boot=500, cluster_ids=None):
    """
    Resample the data with replacement n_boot times, re-run method_func on each
    resample, and collect the resulting driver weights.
    If cluster_ids is provided (e.g. panelist ID), whole panelists are resampled
    together (block/cluster bootstrap) instead of individual rows - this is more
    correct when a person contributes several rows (repeated measures).
    """
    n = len(X)
    boot_rows = []
    for _ in range(n_boot):
        if cluster_ids is not None:
            unique_ids = cluster_ids.unique()
            sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
            idx = np.concatenate([np.where(cluster_ids.values == uid)[0] for uid in sampled_ids])
        else:
            idx = np.random.choice(n, size=n, replace=True)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y.iloc[idx].reset_index(drop=True)
        try:
            w = method_func(Xb, yb)
            boot_rows.append(w)
        except Exception:
            continue
    return pd.DataFrame(boot_rows)


def render_bootstrap_bar(df, value_col, driver_col, direction_col, color_map, title, key_prefix,
                          X, y, method_func, panelist_col, working_df, x_index):
    """Shared UI block: checkbox to toggle bootstrap CIs, then renders either a plain
    bar chart or a bar chart with 95% CI error bars."""
    enable_boot = st.checkbox("🔁 Compute Bootstrap Confidence Intervals", key=f"{key_prefix}_boot")
    if enable_boot:
        n_boot = st.slider("Number of bootstrap resamples", 100, 2000, 500, step=100, key=f"{key_prefix}_nboot")
        cluster_ids = working_df.loc[x_index, panelist_col] if panelist_col != "None" else None
        with st.spinner(f"Running {n_boot} bootstrap resamples..."):
            boot_df = bootstrap_weights(X, y, method_func, n_boot=n_boot, cluster_ids=cluster_ids)
        if boot_df.empty:
            st.warning("Bootstrap did not produce valid resamples (data may be too small/collinear). Showing point estimates only.")
            fig = px.bar(df, x=value_col, y=driver_col, orientation='h', color=direction_col,
                         color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
            return df
        ci_lower = boot_df.quantile(0.025)
        ci_upper = boot_df.quantile(0.975)
        df = df.copy()
        df['CI Lower'] = df[driver_col].map(ci_lower)
        df['CI Upper'] = df[driver_col].map(ci_upper)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df[value_col], y=df[driver_col], orientation='h',
            error_x=dict(type='data', symmetric=False,
                         array=(df['CI Upper'] - df[value_col]).clip(lower=0),
                         arrayminus=(df[value_col] - df['CI Lower']).clip(lower=0)),
            marker_color=df[direction_col].map(color_map)
        ))
        fig.update_layout(title=f"{title} with 95% Bootstrap CI", xaxis_title=value_col, height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Based on {len(boot_df)} successful resamples" +
                   (f", clustered by {panelist_col} (whole panelists resampled together)." if panelist_col != "None" else " (row-level resampling)."))
        return df
    else:
        fig = px.bar(df, x=value_col, y=driver_col, orientation='h', color=direction_col,
                     color_discrete_map=color_map)
        st.plotly_chart(fig, use_container_width=True)
        return df


def apply_product_filter(base_df, product_col, product_choice):
    """Narrows base_df to a single product if one was chosen; returns it unchanged
    for 'All Products', or when no Product ID column has been set in the sidebar."""
    if product_col != "None" and product_choice != "All Products":
        return base_df[base_df[product_col].astype(str) == product_choice]
    return base_df


def product_filter_ui(working_df, product_col, key_prefix):
    """Renders a per-tab 'Run this analysis on: <product>' selector, defaulting to
    'All Products'. Composes with whatever Step 0 sub-target filter is already
    applied to working_df. Shows a prominent warning instead of a selector when no
    Product ID column has been set in the sidebar (Step 1) - this is deliberately
    loud (not a subtle caption) because it's easy to miss otherwise."""
    if product_col == "None":
        st.warning("⚠️ **Product filtering is off** — go to the sidebar → Step 1 → \"Product ID column\" "
                   "and pick the column that identifies which product/fragrance each row belongs to. "
                   "This tab will run on your full sample until you do.")
        return "All Products"
    options = ["All Products"] + sorted(working_df[product_col].dropna().astype(str).unique().tolist())
    return st.selectbox("Run this analysis on:", options, key=f"{key_prefix}_product_filter")


# =============================================================================
# UI APP
# =============================================================================

st.title("📊 Consumer Driver Analysis Suite")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    xl = pd.ExcelFile(uploaded_file)
    selected_sheet = st.selectbox("Select Sheet", xl.sheet_names)
    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Sanitize names immediately
    df = df_raw.copy()
    df.columns = [sanitize_name(c) for c in df.columns]

    # --- STEP 0: SUB-TARGET FILTERING ---
    st.sidebar.header("0. Sub-Target Filtering")
    filter_col = st.sidebar.selectbox("Select Filter Question", ["No Filter"] + list(df.columns))

    working_df = df.copy()
    if filter_col != "No Filter":
        unique_vals = sorted(df[filter_col].dropna().unique().tolist())
        selected_codes = st.sidebar.multiselect(f"Select Codes for {filter_col}", unique_vals)
        if selected_codes:
            working_df = df[df[filter_col].isin(selected_codes)]
            st.sidebar.success(f"Filter applied: {len(working_df)} rows remaining.")
        else:
            st.sidebar.warning("No codes selected: using full sample.")

    # --- STEP 1: VARIABLE SELECTION ---
    st.sidebar.header("1. Variable Selection")
    target = st.sidebar.selectbox("Variable to Explain (Target / Liking)", working_df.columns)

    panelist_col = st.sidebar.selectbox(
        "Panelist ID (optional — enables Mixed-Effects Model & clustered bootstrap)",
        ["None"] + list(working_df.columns)
    )
    product_col = st.sidebar.selectbox(
        "🧴 Product ID column — set this to unlock per-product filtering in EVERY analysis tab below, plus Preference Mapping",
        ["None"] + list(working_df.columns)
    )
    if product_col == "None":
        st.sidebar.warning("⚠️ No Product ID column selected — every analysis tab will run on your full sample "
                            "(no per-product filter will be shown). Pick the column above that identifies which "
                            "product/fragrance each row belongs to if you want to filter by product.")

    st.sidebar.write("Select Explanatory Variables (Drivers / Attributes):")
    available_drivers = [c for c in working_df.columns if c != target]
    selection_df = pd.DataFrame({"Select": [False] * len(available_drivers), "Driver_Variable": available_drivers})

    edited_df = st.sidebar.data_editor(
        selection_df, hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True), "Driver_Variable": st.column_config.TextColumn(disabled=True)},
        use_container_width=True
    )
    features = edited_df[edited_df["Select"] == True]["Driver_Variable"].tolist()

    # --- STEP 2: ANALYSIS SELECTION ---
    st.sidebar.header("2. Analysis Selection")
    analysis_options = [
        "Linear Regression", "RWA", "Shapley Values", "Penalty Analysis (CATA)",
        "JAR Penalty Analysis", "Kano Analysis", "Path Analysis",
        "Mixed-Effects Model", "Preference Mapping"
    ]
    analysis_types = st.sidebar.multiselect("Choose Analyses", analysis_options, default=[], placeholder="Choose options...")

    if target and features and analysis_types:
        data = working_df[[target] + features].dropna()
        y = data[target]
        X = data[features]
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        st.info(f"### 💡 Insights for Sub-Target (N={len(data)})")
        p_values = model.pvalues.iloc[1:]
        significant = p_values[p_values < 0.05].sort_values()
        if not significant.empty:
            for var, pval in significant.items():
                st.markdown(f"- ✅ **{var}** (p-value: {pval:.4f})")
        else:
            st.write("No variables reached significance for this sub-target.")
        st.caption("This summary reflects the Step 0 sub-target filter only, across all products. "
                    "Each tab below can be narrowed further to a single product.")

        st.divider()

        tabs = st.tabs([a for a in analysis_types] + ["Export"])
        results_to_export = {}

        for i, analysis in enumerate(analysis_types):
            with tabs[i]:

                if analysis == "Linear Regression":
                    st.subheader("Linear Regression (Standardized Coefficients)")
                    product_choice = product_filter_ui(working_df, product_col, "linreg")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    tab_X, tab_y = tab_data[features], tab_data[target]

                    if len(tab_data) < len(features) + 2:
                        st.warning(f"⚠️ Not enough data for '{product_choice}' (N={len(tab_data)}) to run this analysis.")
                    else:
                        tab_model = sm.OLS(tab_y, sm.add_constant(tab_X)).fit()
                        std_coefs = tab_model.params.iloc[1:] * (tab_X.std() / tab_y.std())
                        reg_df = pd.DataFrame({'Driver': std_coefs.index, 'Impact Score': std_coefs.values}).sort_values(by='Impact Score', ascending=False)
                        st.plotly_chart(px.bar(reg_df, x='Impact Score', y='Driver', orientation='h', color='Impact Score', color_continuous_scale="RdYlGn"), use_container_width=True)
                        reg_df['Product Filter'] = product_choice
                        results_to_export["Regression"] = reg_df

                elif analysis == "RWA":
                    st.subheader("Relative Weight Analysis (RWA)")
                    product_choice = product_filter_ui(working_df, product_col, "rwa")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    tab_X, tab_y = tab_data[features], tab_data[target]

                    if len(tab_data) < len(features) + 2:
                        st.warning(f"⚠️ Not enough data for '{product_choice}' (N={len(tab_data)}) to run this analysis.")
                    else:
                        rwa_df = run_rwa(tab_X, tab_y)
                        color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': 'gray'}
                        rwa_df = render_bootstrap_bar(
                            rwa_df, 'Weight (%)', 'Driver', 'Direction', color_map,
                            "RWA Weights", "rwa", tab_X, tab_y,
                            lambda Xb, yb: run_rwa(Xb, yb).set_index('Driver')['Weight (%)'],
                            panelist_col, tab_df, tab_X.index
                        )
                        rwa_df['Product Filter'] = product_choice
                        results_to_export["RWA"] = rwa_df

                elif analysis == "Shapley Values":
                    st.subheader("Shapley Values (Contribution to R²)")
                    product_choice = product_filter_ui(working_df, product_col, "shap")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    tab_X, tab_y = tab_data[features], tab_data[target]

                    if len(tab_data) < len(features) + 2:
                        st.warning(f"⚠️ Not enough data for '{product_choice}' (N={len(tab_data)}) to run this analysis.")
                    else:
                        shap_pct, raw_std_coefs = compute_shapley_like(tab_X, tab_y)
                        shap_df = pd.DataFrame({
                            'Driver': features,
                            'Importance (%)': shap_pct.values,
                            'Direction': np.where(raw_std_coefs.values > 0, 'Positive', 'Negative')
                        }).sort_values(by='Importance (%)', ascending=False)
                        color_map = {'Positive': '#2ca02c', 'Negative': '#d62728'}
                        shap_df = render_bootstrap_bar(
                            shap_df, 'Importance (%)', 'Driver', 'Direction', color_map,
                            "Shapley-style Importance", "shap", tab_X, tab_y,
                            lambda Xb, yb: compute_shapley_like(Xb, yb)[0],
                            panelist_col, tab_df, tab_X.index
                        )
                        shap_df['Product Filter'] = product_choice
                        results_to_export["Shapley"] = shap_df

                elif analysis == "Penalty Analysis (CATA)":
                    st.subheader("CATA Penalty Analysis")
                    st.caption("Clusters each attribute by how often consumers check it (reach) and whether checking it "
                               "significantly moves liking up or down (impact) — e.g. a 'Must-Have' is highly checked "
                               "with a confirmed positive impact.")
                    product_choice = product_filter_ui(working_df, product_col, "cata")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    tab_X, tab_y = tab_data[features], tab_data[target]
                    cata_format = st.radio("Data Format", ["0/1", "1/2"], key="cata_radio")
                    reach_threshold_cata = st.slider(
                        "What % checked counts as \"highly checked\" (high reach)?",
                        5, 50, 20, step=5, key="cata_reach"
                    )

                    if tab_data.empty:
                        st.warning(f"⚠️ Not enough data for '{product_choice}' to run this analysis.")
                    else:
                        X_cata = tab_X.copy() - 1 if cata_format == "1/2" else tab_X.copy()
                        pen_list = []
                        for col in features:
                            if 0 in X_cata[col].values and 1 in X_cata[col].values:
                                checked = tab_y[X_cata[col] == 1]
                                unchecked = tab_y[X_cata[col] == 0]
                                diff = checked.mean() - unchecked.mean()
                                if len(checked) > 1 and len(unchecked) > 1:
                                    pval = stats.ttest_ind(checked, unchecked, equal_var=False).pvalue
                                else:
                                    pval = np.nan
                                pen_list.append({
                                    'Attribute': col, 'Impact on Liking': diff,
                                    '% Checked': (X_cata[col].mean() * 100),
                                    'p-value': pval, 'Significant': bool(pval < 0.05) if pd.notna(pval) else False
                                })
                        pen_df = pd.DataFrame(pen_list)

                        if pen_df.empty:
                            st.warning("No attributes look like 0/1 CATA data in this format — check the Data Format setting above.")
                        else:
                            # --- Cluster into a 2 (reach) x 3 (impact, significance-informed) grid ---
                            def classify_cata(row):
                                high_reach = row['% Checked'] >= reach_threshold_cata
                                if row['Significant'] and row['Impact on Liking'] > 0:
                                    return "🟢 Must-Have (Core Strength)" if high_reach else "🔵 Hidden Gem (Opportunity)"
                                elif row['Significant'] and row['Impact on Liking'] < 0:
                                    return "🔴 Red Flag (Liability)" if high_reach else "🟠 Latent Risk (Watch)"
                                else:
                                    return "⚪ Table Stakes (Expected)" if high_reach else "🟣 Low Priority (Unconfirmed)"

                            pen_df['Segment'] = pen_df.apply(classify_cata, axis=1)
                            segment_order = [
                                "🟢 Must-Have (Core Strength)", "🔵 Hidden Gem (Opportunity)",
                                "🔴 Red Flag (Liability)", "🟠 Latent Risk (Watch)",
                                "⚪ Table Stakes (Expected)", "🟣 Low Priority (Unconfirmed)"
                            ]
                            segment_colors = {
                                "🟢 Must-Have (Core Strength)": "#2ecc71", "🔵 Hidden Gem (Opportunity)": "#3498db",
                                "🔴 Red Flag (Liability)": "#e74c3c", "🟠 Latent Risk (Watch)": "#e67e22",
                                "⚪ Table Stakes (Expected)": "#95a5a6", "🟣 Low Priority (Unconfirmed)": "#9b59b6"
                            }
                            pen_df['Segment'] = pd.Categorical(pen_df['Segment'], categories=segment_order, ordered=True)
                            pen_df = pen_df.sort_values(['Segment', '% Checked'], ascending=[True, False]).reset_index(drop=True)

                            # --- Quadrant chart: bubble size = % checked, color = segment ---
                            x_max = max(pen_df['% Checked'].max() * 1.15, reach_threshold_cata * 1.5, 10)
                            y_top = max(pen_df['Impact on Liking'].max() * 1.2, 1)
                            y_bottom = min(pen_df['Impact on Liking'].min() * 1.2, -1)

                            fig = go.Figure()
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.add_vline(x=reach_threshold_cata, line_dash="dash", line_color="gray")

                            for seg in segment_order:
                                sub = pen_df[pen_df['Segment'] == seg]
                                if sub.empty:
                                    continue
                                fig.add_trace(go.Scatter(
                                    x=sub['% Checked'], y=sub['Impact on Liking'], mode='markers+text',
                                    text=sub['Attribute'], textposition='top center',
                                    marker=dict(size=(10 + sub['% Checked'] * 0.6).clip(upper=45), color=segment_colors[seg]),
                                    name=seg
                                ))

                            fig.update_layout(
                                title=f"CATA Penalty Map — {product_choice}",
                                xaxis_title="% of consumers who checked this attribute (bigger bubble = more people)",
                                yaxis_title=f"Impact on {target} (checked vs. not checked)",
                                xaxis_range=[0, x_max], yaxis_range=[y_bottom, y_top],
                                height=550
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander("❓ How to read this chart"):
                                st.markdown(
                                    "- **Each dot** = one attribute from your CATA list.\n"
                                    "- **Further right** = more consumers checked it; **bigger bubble** = the same thing, visually.\n"
                                    "- **Higher up** = liking is higher when consumers check it; below the dashed line means checking it goes with *lower* liking.\n"
                                    "- **Color = segment**, based on reach (left/right of the vertical dashed line) and whether the "
                                    "liking difference is statistically confirmed:\n"
                                    "  - 🟢 **Must-Have**: highly checked, confirmed positive impact — your core strengths, protect these.\n"
                                    "  - 🔵 **Hidden Gem**: not widely checked yet, but confirmed positive impact when it is — an opportunity to make it more noticeable.\n"
                                    "  - 🔴 **Red Flag**: highly checked, confirmed negative impact — your biggest liability, fix first.\n"
                                    "  - 🟠 **Latent Risk**: not widely checked yet, but confirmed negative impact when it is — worth watching in case reach grows.\n"
                                    "  - ⚪ **Table Stakes**: highly checked, but no confirmed effect on liking — expected/assumed, not a differentiator.\n"
                                    "  - 🟣 **Low Priority**: rarely checked and no confirmed effect — safe to deprioritize."
                                )

                            st.markdown(f"#### 🔍 What this means for each attribute — {product_choice}")
                            for _, row in pen_df.iterrows():
                                pct, pval, seg = row['% Checked'], row['p-value'], row['Segment']
                                pval_txt = f"p={pval:.3f}" if pd.notna(pval) else "p=n/a (too few respondents in one group)"
                                if seg == "🟢 Must-Have (Core Strength)":
                                    txt = f"significantly boosts {target} when checked ({pval_txt}). Protect and reinforce this — it's a core strength."
                                elif seg == "🔵 Hidden Gem (Opportunity)":
                                    txt = f"significantly boosts {target} when checked ({pval_txt}), but few consumers notice it yet. Consider making it more salient (formulation intensity, packaging cues, communication)."
                                elif seg == "🔴 Red Flag (Liability)":
                                    txt = f"significantly hurts {target} when checked ({pval_txt}), and a lot of people notice it. This is your clearest priority to fix."
                                elif seg == "🟠 Latent Risk (Watch)":
                                    txt = f"significantly hurts {target} when checked ({pval_txt}), but only a small group notices it today. Lower priority, but worth monitoring."
                                elif seg == "⚪ Table Stakes (Expected)":
                                    txt = f"widely checked, but doesn't move {target} either way ({pval_txt}). It's expected/assumed rather than a differentiator."
                                else:
                                    txt = f"rarely checked and no confirmed effect on {target} ({pval_txt}). Safe to deprioritize."
                                st.markdown(f"{seg} **{row['Attribute']}**: {pct:.0f}% of consumers checked it, and it {txt}")

                            with st.expander("📋 Full statistical detail"):
                                st.dataframe(pen_df.style.format({'% Checked': '{:.1f}', 'Impact on Liking': '{:.3f}', 'p-value': '{:.4f}'}))

                            pen_df['Product Filter'] = product_choice
                            results_to_export["Penalty"] = pen_df

                elif analysis == "JAR Penalty Analysis":
                    st.subheader("JAR (Just-About-Right) Penalty Analysis")
                    st.caption("Select the attributes that were measured on a JAR-type scale (Too Weak ↔ Just About Right ↔ Too Strong).")
                    product_choice = product_filter_ui(working_df, product_col, "jar")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    jar_attrs = st.multiselect("Select JAR-type Attributes", features, key="jar_attrs")
                    scale_type = st.radio(
                        "JAR Scale Format",
                        ["3-point (1=Too Weak, 2=JAR, 3=Too Strong)", "5-point (1-2=Too Weak, 3=JAR, 4-5=Too Strong)"],
                        key="jar_scale"
                    )
                    reach_threshold = st.slider(
                        "What % of consumers counts as \"a lot of people\" for prioritization?",
                        5, 50, 15, step=5, key="jar_reach"
                    )

                    if jar_attrs:
                        jar_results = []
                        for attr in jar_attrs:
                            vals = tab_df[[attr, target]].dropna()
                            if "3-point" in scale_type:
                                weak = vals[vals[attr] == 1]
                                jar = vals[vals[attr] == 2]
                                strong = vals[vals[attr] == 3]
                            else:
                                weak = vals[vals[attr].isin([1, 2])]
                                jar = vals[vals[attr] == 3]
                                strong = vals[vals[attr].isin([4, 5])]

                            n_total = len(vals)
                            if len(jar) == 0 or n_total == 0:
                                continue
                            jar_mean = jar[target].mean()

                            if len(weak) > 1:
                                p_weak = stats.ttest_ind(jar[target], weak[target], equal_var=False).pvalue
                                drop_weak = jar_mean - weak[target].mean()
                            else:
                                p_weak, drop_weak = np.nan, np.nan

                            if len(strong) > 1:
                                p_strong = stats.ttest_ind(jar[target], strong[target], equal_var=False).pvalue
                                drop_strong = jar_mean - strong[target].mean()
                            else:
                                p_strong, drop_strong = np.nan, np.nan

                            jar_results.append({
                                'Attribute': attr, 'Direction': 'Too Weak',
                                '% Selecting': len(weak) / n_total * 100, 'Impact on Liking': drop_weak,
                                'p-value': p_weak, 'Significant': bool(p_weak < 0.05) if pd.notna(p_weak) else False
                            })
                            jar_results.append({
                                'Attribute': attr, 'Direction': 'Too Strong',
                                '% Selecting': len(strong) / n_total * 100, 'Impact on Liking': drop_strong,
                                'p-value': p_strong, 'Significant': bool(p_strong < 0.05) if pd.notna(p_strong) else False
                            })

                        jar_df = pd.DataFrame(jar_results).dropna(subset=['Impact on Liking'])
                        if not jar_df.empty:

                            # --- Plain-language verdict per row ---
                            def classify(row):
                                if row['Impact on Liking'] <= 0:
                                    return "🟢 Not a concern"
                                elif row['Significant'] and row['% Selecting'] >= reach_threshold:
                                    return "🔴 Priority fix"
                                elif row['Significant'] and row['% Selecting'] < reach_threshold:
                                    return "🟠 Real, but niche"
                                elif (not row['Significant']) and row['% Selecting'] >= reach_threshold:
                                    return "🟡 Worth watching"
                                else:
                                    return "⚪ Low priority"

                            jar_df['Verdict'] = jar_df.apply(classify, axis=1)
                            verdict_order = ["🔴 Priority fix", "🟠 Real, but niche", "🟡 Worth watching", "⚪ Low priority", "🟢 Not a concern"]
                            verdict_colors = {
                                "🔴 Priority fix": "#e74c3c", "🟠 Real, but niche": "#e67e22",
                                "🟡 Worth watching": "#f1c40f", "⚪ Low priority": "#bdc3c7",
                                "🟢 Not a concern": "#2ecc71"
                            }
                            jar_df['Verdict'] = pd.Categorical(jar_df['Verdict'], categories=verdict_order, ordered=True)
                            jar_df = jar_df.sort_values(['Verdict', '% Selecting'], ascending=[True, False]).reset_index(drop=True)

                            # --- Chart: bubble size = % affected, color = confidence/priority ---
                            x_max = max(jar_df['% Selecting'].max() * 1.15, reach_threshold * 1.5, 10)
                            y_top = max(jar_df['Impact on Liking'].max() * 1.2, 5)
                            y_bottom = min(jar_df['Impact on Liking'].min() * 1.2, -5)

                            fig = go.Figure()
                            fig.add_shape(type="rect", x0=reach_threshold, x1=x_max, y0=0, y1=y_top,
                                          fillcolor="rgba(231,76,60,0.08)", line_width=0, layer="below")
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.add_vline(x=reach_threshold, line_dash="dash", line_color="gray")

                            for verdict in verdict_order:
                                sub = jar_df[jar_df['Verdict'] == verdict]
                                if sub.empty:
                                    continue
                                fig.add_trace(go.Scatter(
                                    x=sub['% Selecting'], y=sub['Impact on Liking'], mode='markers+text',
                                    text=sub['Attribute'] + " (" + sub['Direction'].astype(str) + ")",
                                    textposition='top center',
                                    marker=dict(size=(10 + sub['% Selecting'] * 0.6).clip(upper=45), color=verdict_colors[verdict]),
                                    name=verdict
                                ))

                            fig.update_layout(
                                title=f"JAR Penalty Chart — {product_choice} — where to focus first",
                                xaxis_title="% of consumers who said this (bigger bubble = more people)",
                                yaxis_title=f"Impact on {target} (higher = hurts liking more)",
                                xaxis_range=[0, x_max], yaxis_range=[y_bottom, y_top],
                                height=550
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander("❓ How to read this chart"):
                                st.markdown(
                                    "- **Each dot** = one attribute in one direction (e.g. *too strong*).\n"
                                    "- **Further right** = more consumers said this; **bigger bubble** = the same thing, visually.\n"
                                    "- **Higher up** = the more it drags liking scores down; below the dashed line means that group actually liked it just as much or more.\n"
                                    "- **Shaded red zone** (top-right) = affects a lot of people *and* hurts liking — your best candidates to act on.\n"
                                    "- **Color** = how confident we are it's real: red/orange are statistically confirmed, yellow is promising but needs a bigger sample to be sure, gray/green are low priority."
                                )

                            st.markdown(f"#### 🔍 What this means for each attribute — {product_choice}")
                            for _, row in jar_df.iterrows():
                                direction_phrase = "too weak" if row['Direction'] == 'Too Weak' else "too strong"
                                pct, pval = row['% Selecting'], row['p-value']
                                if row['Verdict'] == "🔴 Priority fix":
                                    txt = (f"significantly pulls down {target} (p={pval:.3f}). "
                                           f"This is your clearest, most confident opportunity to act on.")
                                elif row['Verdict'] == "🟠 Real, but niche":
                                    txt = (f"the effect is statistically real (p={pval:.3f}), but it only affects a small "
                                           f"share of consumers — lower priority unless that group matters strategically.")
                                elif row['Verdict'] == "🟡 Worth watching":
                                    txt = (f"liking looks lower for this group, but with the current sample it isn't "
                                           f"statistically confirmed yet (p={pval:.3f}). Worth a bigger sample or a follow-up test before reformulating.")
                                elif row['Verdict'] == "⚪ Low priority":
                                    txt = f"it's a small group and not statistically confirmed (p={pval:.3f}) — safe to deprioritize."
                                else:
                                    txt = "this group doesn't actually like the product any less for it — no action needed."
                                st.markdown(f"{row['Verdict']} **{row['Attribute']} — {row['Direction']}**: {pct:.0f}% of consumers say it's {direction_phrase}, and {txt}")

                            with st.expander("📋 Full statistical detail"):
                                st.dataframe(jar_df.style.format({'% Selecting': '{:.1f}', 'Impact on Liking': '{:.3f}', 'p-value': '{:.4f}'}))

                            jar_df['Product Filter'] = product_choice
                            results_to_export["JAR_Penalty"] = jar_df
                        else:
                            st.warning(f"Not enough data for '{product_choice}' in the JAR categories to compute penalties.")
                    else:
                        st.info("Select at least one JAR-type attribute above to run this analysis.")

                elif analysis == "Kano Analysis":
                    st.subheader("Kano Strategic Classification")
                    product_choice = product_filter_ui(working_df, product_col, "kano")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    tab_X, tab_y = tab_data[features], tab_data[target]

                    if tab_data.empty:
                        st.warning(f"⚠️ Not enough data for '{product_choice}' to run this analysis.")
                    else:
                        kano_list = []
                        for col in features:
                            reward = tab_y[tab_X[col] >= tab_X[col].median()].mean() - tab_y.mean()
                            penalty = tab_y.mean() - tab_y[tab_X[col] < tab_X[col].median()].mean()

                            if reward > penalty and reward > 0.1:
                                cat = "Delighter (Attractive)"
                            elif penalty > reward and penalty > 0.1:
                                cat = "Must-have (Basic)"
                            elif abs(reward - penalty) < 0.1 and reward > 0.1:
                                cat = "Linear (Performance)"
                            else:
                                cat = "Indifferent"

                            kano_list.append({'Driver': col, 'Reward Potential': reward, 'Penalty Potential': penalty, 'Category': cat})

                        kano_df = pd.DataFrame(kano_list)
                        st.plotly_chart(px.scatter(kano_df, x='Penalty Potential', y='Reward Potential', color='Category', text='Driver', title=f"Kano Map — {product_choice}"), use_container_width=True)
                        st.caption("Note: this is a proxy classification based on a median split of each driver, not the full Kano method (which requires paired functional/dysfunctional questions).")
                        st.table(kano_df)
                        kano_df['Product Filter'] = product_choice
                        results_to_export["Kano"] = kano_df

                elif analysis == "Path Analysis":
                    st.subheader("Path Analysis (SEM)")
                    product_choice = product_filter_ui(working_df, product_col, "path")
                    tab_df = apply_product_filter(working_df, product_col, product_choice)
                    tab_data = tab_df[[target] + features].dropna()
                    path_syntax = st.text_area("Syntax", value=f"{target} ~ {' + '.join(features)}", key="path_syntax")
                    if st.button("Run Path Model"):
                        if tab_data.empty:
                            st.warning(f"⚠️ Not enough data for '{product_choice}' to run this analysis.")
                        else:
                            try:
                                sem = Model(path_syntax)
                                sem.fit(tab_data)
                                res = sem.inspect()
                                paths = res[res['op'] == '~']
                                labels = list(set(paths['lval'].tolist() + paths['rval'].tolist()))
                                fig = go.Figure(data=[go.Sankey(
                                    node=dict(pad=15, thickness=20, label=labels, color="blue"),
                                    link=dict(source=[labels.index(x) for x in paths['rval']],
                                              target=[labels.index(x) for x in paths['lval']],
                                              value=np.abs(paths['Estimate']).tolist(),
                                              label=paths['Estimate'].round(3).astype(str).tolist()))])
                                st.plotly_chart(fig, use_container_width=True)
                                res['Product Filter'] = product_choice
                                results_to_export["Path"] = res
                            except Exception as e:
                                st.error(f"SEM Error: {e}")

                elif analysis == "Mixed-Effects Model":
                    st.subheader("Mixed-Effects Model (Random Intercept per Panelist)")
                    st.caption("Accounts for repeated measures — the same consumer rating several fragrances — by giving each panelist their own baseline liking level.")
                    product_choice = product_filter_ui(working_df, product_col, "mixed")
                    if product_choice != "All Products":
                        st.caption("ℹ️ Mixed-effects models rely on the same panelist rating several products. Filtering to a single product "
                                   "usually leaves only one rating per panelist, which can make the random panelist effect hard to estimate reliably.")

                    if panelist_col == "None":
                        st.warning("⚠️ Please select a Panelist ID column in the sidebar (Step 1) to run this analysis.")
                    else:
                        tab_df = apply_product_filter(working_df, product_col, product_choice)
                        mm_data = tab_df[[target, panelist_col] + features].dropna()

                        if len(mm_data) < len(features) + 2 or mm_data[panelist_col].nunique() < 2:
                            st.warning(f"⚠️ Not enough data for '{product_choice}' to fit a mixed model (need ratings from at least 2 panelists).")
                        else:
                            mm_std = mm_data.copy()
                            mm_std[features] = (mm_data[features] - mm_data[features].mean()) / mm_data[features].std()
                            mm_std[target] = (mm_data[target] - mm_data[target].mean()) / mm_data[target].std()

                            formula = f"{target} ~ {' + '.join(features)}"
                            try:
                                mixed_model = smf.mixedlm(formula, mm_std, groups=mm_std[panelist_col])
                                mixed_result = mixed_model.fit()

                                fe = mixed_result.fe_params.drop('Intercept', errors='ignore')
                                pvals = mixed_result.pvalues
                                mm_df = pd.DataFrame({
                                    'Driver': fe.index,
                                    'Standardized Coefficient': fe.values,
                                    'p-value': [pvals.get(d, np.nan) for d in fe.index]
                                }).sort_values(by='Standardized Coefficient', ascending=False)

                                st.plotly_chart(px.bar(mm_df, x='Standardized Coefficient', y='Driver', orientation='h',
                                                        color='Standardized Coefficient', color_continuous_scale="RdYlGn",
                                                        title=f"Mixed Model Fixed Effects (Standardized) — {product_choice}"), use_container_width=True)

                                n_panelists = mm_data[panelist_col].nunique()
                                st.caption(f"Random intercept fit across {n_panelists} panelists ({len(mm_data)} total ratings).")
                                st.dataframe(mm_df.style.format({'Standardized Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))

                                with st.expander("Full Model Summary"):
                                    st.text(str(mixed_result.summary()))

                                mm_df['Product Filter'] = product_choice
                                results_to_export["MixedModel"] = mm_df
                            except Exception as e:
                                st.error(f"Mixed model failed to converge: {e}")

                elif analysis == "Preference Mapping":
                    st.subheader("Preference Mapping (PCA + External Vector Model)")
                    st.caption("This technique compares several products at once, so there's no single-product filter here — "
                               "it always uses every product available after the Step 0 sub-target filter.")
                    if product_col == "None":
                        st.warning("⚠️ Please select a Product ID column in the sidebar (Step 1) to run Preference Mapping.")
                    elif len(features) < 2:
                        st.warning("⚠️ Select at least 2 attributes as drivers to build a 2D map.")
                    else:
                        pm_data = working_df[[product_col, target] + features].dropna()
                        product_means = pm_data.groupby(product_col)[features].mean()
                        liking_means = pm_data.groupby(product_col)[target].mean()

                        if len(product_means) < 3:
                            st.warning("⚠️ Need at least 3 distinct products for a meaningful preference map.")
                        else:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(product_means)
                            pca = PCA(n_components=2)
                            scores = pca.fit_transform(X_scaled)
                            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                            var_exp = pca.explained_variance_ratio_ * 100

                            scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2'], index=product_means.index)
                            loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)

                            # External preference vector: regress mean liking per product on PC scores
                            # (pass pandas objects, not .values, so .params comes back as a labeled
                            # Series with .loc access rather than a bare numpy array)
                            pref_exog = sm.add_constant(scores_df, has_constant='add')
                            pref_model = sm.OLS(liking_means, pref_exog).fit()
                            vec_pc1, vec_pc2 = pref_model.params.loc['PC1'], pref_model.params.loc['PC2']

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=scores_df['PC1'], y=scores_df['PC2'], mode='markers+text',
                                text=scores_df.index.astype(str), textposition='top center',
                                marker=dict(size=14, color='#1f77b4'), name='Products'
                            ))

                            scale_factor = np.abs(scores_df.values).max() / max(np.abs(loadings_df.values).max(), 1e-9) * 0.8
                            for attr in loadings_df.index:
                                fig.add_trace(go.Scatter(
                                    x=[0, loadings_df.loc[attr, 'PC1'] * scale_factor],
                                    y=[0, loadings_df.loc[attr, 'PC2'] * scale_factor],
                                    mode='lines+text', text=[None, attr],
                                    line=dict(color='gray', width=1), showlegend=False
                                ))

                            vec_norm = max(np.hypot(vec_pc1, vec_pc2), 1e-9)
                            liking_scale = np.abs(scores_df.values).max() / vec_norm * 0.9
                            fig.add_trace(go.Scatter(
                                x=[0, vec_pc1 * liking_scale], y=[0, vec_pc2 * liking_scale],
                                mode='lines+text', text=[None, f'{target} (Liking)'],
                                line=dict(color='red', width=3), name='Liking Vector'
                            ))

                            fig.update_layout(
                                title="Preference Map — Products, Attribute Loadings & Liking Vector",
                                xaxis_title=f"PC1 ({var_exp[0]:.1f}% variance)",
                                yaxis_title=f"PC2 ({var_exp[1]:.1f}% variance)",
                                height=650
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Red arrow = direction of increasing average liking. Gray arrows = attribute loadings. "
                                       "Products lying further along the red arrow are, on average, more liked; attributes pointing "
                                       "the same way as the red arrow are generally liking-positive.")

                            results_to_export["PrefMap_Scores"] = scores_df
                            results_to_export["PrefMap_Loadings"] = loadings_df

        with tabs[-1]:
            st.subheader("Download Results")
            if results_to_export:
                st.caption("Each sheet includes a 'Product Filter' column showing which product (or 'All Products') that tab was set to when exported.")
                xlsx_data = to_excel(results_to_export)
                st.download_button("📥 Download Analysis (.xlsx)", xlsx_data, "subtarget_analysis.xlsx")
            else:
                st.info("Run at least one analysis to enable export.")
    else:
        st.info("👈 Complete the sidebar steps to begin.")
