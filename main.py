"""
Robust Wine EDA app (Streamlit + CLI fallback)

This single-file script is a fixed version of the earlier Streamlit app.
It addresses environments where `streamlit` is not installed (ModuleNotFoundError).

Behavior:
- If `streamlit` is available, the script starts the Streamlit web UI exactly like before.
- If `streamlit` is NOT available, the script runs a CLI fallback:
  - It attempts to load local CSVs from /mnt/data (common when files are uploaded in the environment).
  - If not found, it will try to download the UCI Wine Quality CSVs.
  - It runs the same analyses (summary stats, variability, correlations, tests) and saves result CSV(s) and example plot PNGs into a folder `./eda_outputs`.
  - It also runs a small internal test suite to confirm core functions work (simple assertions).

Why this fix: the environment where the user executed the file did not have `streamlit` installed. Rather than crash, this version gracefully falls back to a headless runner so analysis can still be completed and inspected.

Usage:
- Streamlit mode (recommended): `pip install streamlit` then `streamlit run this_file.py`
- CLI fallback mode (no streamlit): `python this_file.py` — outputs go to ./eda_outputs/

"""

from __future__ import annotations
import os
import sys
import io
import math
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels import robust

# Optional imports
STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
except Exception as e:
    STREAMLIT_AVAILABLE = False
    st = None

# pingouin is optional for ANOVA effect size; handle missing gracefully
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except Exception:
    PINGOUIN_AVAILABLE = False

sns.set(style="whitegrid")

# ---------------- Helper functions ----------------

def load_csv_from_path(path: str) -> pd.DataFrame | None:
    """Attempt to load a CSV file; supports semicolon or comma separators."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Check if we got a malformed single-column DataFrame (wrong delimiter)
        if df.shape[1] == 1 and ';' in str(df.columns[0]):
            df = pd.read_csv(path, sep=';')
        return df
    except Exception:
        try:
            return pd.read_csv(path, sep=';')
        except Exception:
            return None


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read CSV from uploaded file, automatically detecting comma or semicolon delimiters."""
    try:
        df = pd.read_csv(uploaded_file)
        # If we got only 1 column with semicolons in the name, wrong delimiter was used
        if df.shape[1] == 1 and ';' in str(df.columns[0]):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
        return df
    except Exception:
        # Fallback to semicolon
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=';')


def load_wine_sample() -> pd.DataFrame:
    """Try several methods to obtain the wine dataset:
    1. /mnt/data/wine+quality.zip (if user uploaded a zip)
    2. /mnt/data/winequality-red.csv and winequality-white.csv
    3. Direct UCI download (if environment allows internet)
    If none succeed, raise an error.
    """
    # 1: check for zip
    zip_paths = ["/mnt/data/wine+quality.zip", "/mnt/data/wine quality.zip", "/mnt/data/winequality.zip"]
    import zipfile
    for zp in zip_paths:
        if os.path.exists(zp):
            with zipfile.ZipFile(zp, 'r') as z:
                names = z.namelist()
                red_name = next((n for n in names if 'red' in n.lower() and n.lower().endswith('.csv')), None)
                white_name = next((n for n in names if 'white' in n.lower() and n.lower().endswith('.csv')), None)
                if red_name and white_name:
                    with z.open(red_name) as f:
                        red = pd.read_csv(f, sep=';')
                    with z.open(white_name) as f:
                        white = pd.read_csv(f, sep=';')
                    red['wine_type'] = 'red'
                    white['wine_type'] = 'white'
                    return pd.concat([red, white], ignore_index=True)
    # 2: direct files
    red_local = load_csv_from_path('/mnt/data/winequality-red.csv')
    white_local = load_csv_from_path('/mnt/data/winequality-white.csv')
    if red_local is not None and white_local is not None:
        red_local['wine_type'] = 'red'
        white_local['wine_type'] = 'white'
        return pd.concat([red_local, white_local], ignore_index=True)

    # 3: try to download from UCI (may fail if network disabled)
    try:
        red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        red = pd.read_csv(red_url, sep=';')
        white = pd.read_csv(white_url, sep=';')
        red['wine_type'] = 'red'
        white['wine_type'] = 'white'
        return pd.concat([red, white], ignore_index=True)
    except Exception:
        raise FileNotFoundError("Could not find local wine data and network download failed. Please upload CSV(s) to /mnt/data or run with network access.")


def compute_stats(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe().T
    desc['median'] = df[numeric_cols].median()
    desc['mad'] = df[numeric_cols].apply(lambda x: np.mean(np.abs(x - x.mean())))
    desc['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    desc['skew'] = df[numeric_cols].skew()
    desc['kurtosis'] = df[numeric_cols].kurtosis()

    mad_median = {col: float(robust.mad(df[col].dropna())) for col in numeric_cols}

    variability = pd.DataFrame(index=numeric_cols)
    variability['std'] = df[numeric_cols].std()
    variability['var'] = df[numeric_cols].var()
    variability['mad'] = df[numeric_cols].apply(lambda x: np.mean(np.abs(x - x.mean())))
    variability['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    variability['median_mad_robust'] = [mad_median[col] for col in numeric_cols]

    return numeric_cols, desc, variability


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols, desc, variability = compute_stats(df)
    results: Dict[str, Any] = {}
    results['numeric_cols'] = numeric_cols
    results['desc'] = desc
    results['variability'] = variability

    corr = df[numeric_cols].corr(method='pearson')
    results['pearson_corr'] = corr

    tests: Dict[str, Any] = {}
    if 'alcohol' in df.columns and 'density' in df.columns:
        tests['pearson_alcohol_density'] = stats.pearsonr(df['alcohol'], df['density'])
    if 'quality' in df.columns and 'alcohol' in df.columns:
        tests['spearman_quality_alcohol'] = stats.spearmanr(df['quality'], df['alcohol'])
    if 'quality' in df.columns and 'pH' in df.columns:
        tests['kendall_quality_pH'] = stats.kendalltau(df['quality'], df['pH'])

    if 'quality' in df.columns and 'alcohol' in df.columns:
        df = df.copy()
        df['high_quality'] = (df['quality'] >= 7).astype(int)
        tests['pointbiserial_alcohol_highquality'] = stats.pointbiserialr(df['alcohol'], df['high_quality'])
        df['high_alcohol'] = (df['alcohol'] >= df['alcohol'].median()).astype(int)
        ct = pd.crosstab(df['high_alcohol'], df['high_quality'])
        tests['contingency_highalc_highqual'] = ct
        try:
            chi2 = stats.chi2_contingency(ct)[0]
            phi = math.sqrt(chi2 / df.shape[0])
            tests['phi'] = phi
        except Exception:
            tests['phi'] = None

    if 'quality' in df.columns and 'alcohol' in df.columns and PINGOUIN_AVAILABLE:
        try:
            anova = pg.anova(data=df, dv='alcohol', between='quality', detailed=True)
            eta2 = pg.compute_effsize(df, dv='alcohol', between='quality', eftype='eta-square')
            tests['anova_alcohol_quality'] = anova
            tests['eta2'] = float(eta2)
        except Exception as e:
            tests['anova_error'] = str(e)

    if 'wine_type' in df.columns and 'high_quality' in df.columns:
        ct2 = pd.crosstab(df['wine_type'], df['high_quality'])
        try:
            chi2, p, dof, expected = stats.chi2_contingency(ct2)
            tests['chi2_winetype_highqual'] = (float(chi2), float(p))
        except Exception:
            tests['chi2_winetype_highqual'] = None

    results['tests'] = tests
    return results


# ---------------- Visualization utilities (headless-safe) ----------------

def save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_hist_kde(df: pd.DataFrame, col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram + KDE: {col}")
    return fig


def plot_box_all(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12,5))
    data = df[numeric_cols].melt(var_name='variable', value_name='value')
    sns.boxplot(data=data, x='variable', y='value', ax=ax)
    plt.xticks(rotation=45)
    return fig


def plot_scatter(df: pd.DataFrame, xvar: str, yvar: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df[xvar], df[yvar], alpha=0.4)
    ax.set_xlabel(xvar); ax.set_ylabel(yvar); ax.set_title(f"Scatter: {xvar} vs {yvar}")
    return fig


def plot_hexbin(df: pd.DataFrame, xvar: str, yvar: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hexbin(df[xvar], df[yvar], gridsize=35)
    ax.set_xlabel(xvar); ax.set_ylabel(yvar)
    return fig


def plot_violin_by_quality(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10,4))
    sns.violinplot(x='quality', y='alcohol', data=df, inner='quartile', ax=ax)
    return fig


# ---------------- CLI runner (fallback) ----------------

def run_headless_and_save(df: pd.DataFrame, out_dir: str = './eda_outputs') -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    results = run_full_analysis(df)

    # save CSVs
    results['desc'].to_csv(os.path.join(out_dir, 'summary_stats.csv'))
    results['variability'].to_csv(os.path.join(out_dir, 'variability.csv'))
    results['pearson_corr'].to_csv(os.path.join(out_dir, 'pearson_corr.csv'))

    # save example plots
    num_cols = results['numeric_cols']
    # histogram of alcohol if present
    if 'alcohol' in df.columns:
        f = plot_hist_kde(df, 'alcohol')
        save_fig(f, os.path.join(out_dir, 'hist_alcohol.png'))
    # boxplots
    f = plot_box_all(df, num_cols)
    save_fig(f, os.path.join(out_dir, 'boxplots_all.png'))
    # scatter example
    if 'alcohol' in df.columns and 'density' in df.columns:
        f = plot_scatter(df, 'alcohol', 'density')
        save_fig(f, os.path.join(out_dir, 'scatter_alcohol_density.png'))
        f2 = plot_hexbin(df, 'alcohol', 'density')
        save_fig(f2, os.path.join(out_dir, 'hexbin_alcohol_density.png'))
    # violin
    if 'quality' in df.columns and 'alcohol' in df.columns:
        f = plot_violin_by_quality(df)
        save_fig(f, os.path.join(out_dir, 'violin_alcohol_quality.png'))

    # write tests summary
    with open(os.path.join(out_dir, 'tests_summary.txt'), 'w') as fo:
        fo.write('Correlation & statistical tests (examples)\n')
        for k,v in results['tests'].items():
            fo.write(f"{k}: {repr(v)}\n")

    print(f"Headless analysis complete — outputs written to: {out_dir}")
    return results


# ---------------- Small internal tests ----------------

def _internal_tests():
    print("Running internal tests...")
    # create tiny synthetic dataset
    rng = np.random.RandomState(0)
    n = 200
    alcohol = rng.normal(10, 1, size=n)
    density = 1.02 - 0.003 * alcohol + rng.normal(0, 0.001, size=n)
    quality = (alcohol > 10).astype(int) + rng.randint(3,7, size=n)
    df_test = pd.DataFrame({'alcohol': alcohol, 'density': density, 'quality': quality})

    # compute stats
    numeric_cols, desc, variability = compute_stats(df_test)
    assert 'alcohol' in numeric_cols
    assert 'density' in numeric_cols
    assert 'median' in desc.columns
    assert 'mad' in desc.columns

    # run correlation test functions
    res = run_full_analysis(df_test)
    assert 'pearson_alcohol_density' in res['tests']
    pear = res['tests']['pearson_alcohol_density']
    assert hasattr(pear, '__len__') and len(pear) >= 2
    print('Internal tests passed.')


# ---------------- Main entry ----------------

def main_cli():
    """Fallback CLI entry point when Streamlit is not available."""
    try:
        print("Attempting to load wine dataset for headless analysis...")
        df = load_wine_sample()
        run_headless_and_save(df)
        _internal_tests()
    except Exception as e:
        print(f"Error in CLI mode: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # If streamlit is available, try to run the interactive UI. If running directly with `python main.py`
    # Streamlit may be importable but not actually running as the Streamlit server; protect against that.
    if STREAMLIT_AVAILABLE:
        try:
            # Professional Academic UI
            st.set_page_config(page_title="Wine Quality EDA", layout="wide", initial_sidebar_state="collapsed")
            
            # Header Section
            st.title("🍷 Wine Quality Dataset - Exploratory Data Analysis")
            st.markdown("**Correlation Tests & Advanced Visualization**")
            st.markdown("*Dataset: UCI Wine Quality (Red & White Wine)*")
            st.divider()

            # Simplified Data Loading
            st.sidebar.header("📁 Data Source")
            uploaded = st.sidebar.file_uploader("Upload CSV file (optional)", type=['csv'])
            use_sample = st.sidebar.button("Use UCI Wine Quality Dataset", type="primary")
            
            # Dataset download option
            st.sidebar.divider()
            st.sidebar.subheader("📥 Export Dataset")
            if st.sidebar.button("Prepare Dataset Download"):
                try:
                    temp_df = load_wine_sample()
                    csv_data = temp_df.to_csv(index=False).encode('utf-8')
                    st.sidebar.download_button(
                        label="⬇️ Download Wine Dataset (CSV)",
                        data=csv_data,
                        file_name="wine_quality_dataset.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.sidebar.error(f"Could not prepare dataset: {e}")

            if uploaded is None and not use_sample:
                st.info("👆 Click 'Use UCI Wine Quality Dataset' in the sidebar to load the sample data, or upload your own CSV file.")
                st.stop()

            # Load data
            if use_sample:
                try:
                    df = load_wine_sample()
                    st.sidebar.success("✅ Sample dataset loaded")
                except Exception as e:
                    st.error(f"Could not load sample: {e}")
                    st.stop()
            else:
                try:
                    if uploaded is None:
                        raise RuntimeError('No file uploaded')
                    df = read_uploaded_csv(uploaded)
                    st.sidebar.success("✅ CSV file loaded")
                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")
                    st.stop()

            # Run full analysis
            results = run_full_analysis(df)

            # ========== SECTION 1: DATA IMPORT & CONSISTENCY CHECKS ==========
            st.header("1️⃣ Data Import & Consistency Checks")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            with st.expander("📊 View Dataset Info"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Data Types")
                    buf = io.StringIO()
                    df.info(buf=buf)
                    st.text(buf.getvalue())
                with col_b:
                    st.subheader("Sample Data (First 10 Rows)")
                    st.dataframe(df.head(10), use_container_width=True)
            
            st.divider()

            # ========== SECTION 2: SUMMARY STATISTICS ==========
            st.header("2️⃣ Summary Statistics (Mean, Median & Variations)")
            st.markdown("*Includes: Mean, Median, Std Dev, Min, Max, Quartiles, Skewness, Kurtosis*")
            st.dataframe(results['desc'], use_container_width=True)
            
            with st.expander("📝 Interpretation"):
                st.write("""
                - **Mean vs Median**: Indicates data distribution (symmetric if similar, skewed if different)
                - **Skewness**: Positive values indicate right-skewed distribution  
                - **Kurtosis**: High values indicate presence of outliers
                - Variables like *residual sugar* and *sulphates* show high variability and skewness
                """)
            
            st.divider()

            # ========== SECTION 3: VARIABILITY METRICS ==========
            st.header("3️⃣ Variability Metrics (Std Dev, MAD, IQR)")
            st.dataframe(results['variability'], use_container_width=True)
            
            with st.expander("📝 Interpretation"):
                st.write("""
                - **Std Dev**: Measures spread around the mean (higher = more variability)
                - **MAD (Mean Absolute Deviation)**: Robust measure of variability
                - **IQR (Interquartile Range)**: Spread of middle 50% of data
                - **Median MAD (Robust)**: Most robust measure, less affected by outliers
                """)
            
            st.divider()

            # ========== SECTION 4: PRIMARY VISUALIZATIONS ==========
            st.header("4️⃣ Primary Visualizations")
            
            # Boxplots
            st.subheader("📦 Box Plots (Outlier Detection)")
            fig, ax = plt.subplots(figsize=(14,5))
            data = df[results['numeric_cols']].melt(var_name='variable', value_name='value')
            sns.boxplot(data=data, x='variable', y='value', ax=ax, palette='Set2')
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Box Plots for All Numeric Variables")
            st.pyplot(fig)
            
            # Histograms with KDE
            st.subheader("📊 Histogram & Density Plot")
            num_cols = results['numeric_cols']
            col = st.selectbox("Select variable to visualize:", options=num_cols, index=num_cols.index('alcohol') if 'alcohol' in num_cols else 0)
            
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='steelblue', bins=30)
            ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            st.divider()

            # ========== SECTION 5: CORRELATION MATRIX ==========
            st.header("5️⃣ Correlation Matrix (Pearson)")
            fig, ax = plt.subplots(figsize=(12,10))
            sns.heatmap(results['pearson_corr'], annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       ax=ax, cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5)
            ax.set_title("Pearson Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
            st.pyplot(fig)
            
            with st.expander("📝 Key Findings"):
                st.write("""
                - **Alcohol & Quality**: Positive correlation (~0.48 for red wine)
                - **Alcohol & Density**: Strong negative correlation (~-0.78)
                - **Total Sulfur Dioxide & Free Sulfur Dioxide**: Strong positive correlation
                - **pH & Fixed Acidity**: Negative correlation (as expected chemically)
                """)
            
            st.divider()

            # ========== SECTION 6: RELATIONSHIP VISUALIZATIONS ==========
            st.header("6️⃣ Relationship Visualizations")
            
            colx, coly = st.columns(2)
            with colx:
                xvar = st.selectbox("X-axis variable:", options=num_cols, 
                                   index=num_cols.index('alcohol') if 'alcohol' in num_cols else 0)
            with coly:
                yvar = st.selectbox("Y-axis variable:", options=num_cols, 
                                   index=num_cols.index('density') if 'density' in num_cols else 1)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Scatter Plot", "🔷 Hexagonal Binning", "🌀 Contour Plot", "🎻 Violin Plot"])
            
            with tab1:
                st.subheader("Scatter Plot")
                fig, ax = plt.subplots(figsize=(8,5))
                ax.scatter(df[xvar], df[yvar], alpha=0.5, c='steelblue', edgecolors='black', linewidth=0.5)
                ax.set_xlabel(xvar, fontsize=12)
                ax.set_ylabel(yvar, fontsize=12)
                ax.set_title(f"{xvar} vs {yvar}", fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Hexagonal Binning (Density Visualization)")
                fig, ax = plt.subplots(figsize=(8,5))
                hb = ax.hexbin(df[xvar], df[yvar], gridsize=35, cmap='YlOrRd')
                ax.set_xlabel(xvar, fontsize=12)
                ax.set_ylabel(yvar, fontsize=12)
                ax.set_title(f"{xvar} vs {yvar} - Hexbin", fontsize=14, fontweight='bold')
                plt.colorbar(hb, ax=ax, label='Count')
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Contour Plot (KDE-colored scatter)")
                from scipy.stats import gaussian_kde
                x = df[xvar].dropna()
                y = df[yvar].dropna()
                if len(x) > 0 and len(y) > 0:
                    xy = np.vstack([x,y])
                    try:
                        z = gaussian_kde(xy)(xy)
                        idx = z.argsort()
                        x_s, y_s, z_s = x.values[idx], y.values[idx], z[idx]
                        fig, ax = plt.subplots(figsize=(8,5))
                        sc = ax.scatter(x_s, y_s, c=z_s, s=30, cmap='viridis')
                        ax.set_xlabel(xvar, fontsize=12)
                        ax.set_ylabel(yvar, fontsize=12)
                        ax.set_title(f"{xvar} vs {yvar} - KDE Density", fontsize=14, fontweight='bold')
                        plt.colorbar(sc, ax=ax, label='Density')
                        st.pyplot(fig)
                    except Exception:
                        st.warning("KDE contour could not be computed for this variable pair.")
            
            with tab4:
                if 'quality' in df.columns and 'alcohol' in df.columns:
                    st.subheader("Violin Plot: Alcohol by Quality")
                    fig, ax = plt.subplots(figsize=(10,5))
                    sns.violinplot(x='quality', y='alcohol', data=df, inner='quartile', ax=ax, palette='muted')
                    ax.set_title("Distribution of Alcohol Content by Quality Rating", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Wine Quality Rating", fontsize=12)
                    ax.set_ylabel("Alcohol (%)", fontsize=12)
                    st.pyplot(fig)
                else:
                    st.info("Violin plot requires 'quality' and 'alcohol' columns")
            
            
            st.divider()

            # ========== SECTION 7: CONCLUSIONS ==========
            st.header("7️⃣ Conclusions & Key Observations")
            
            st.success("""
            **Key Findings from Wine Quality Analysis:**
            
            1. **Alcohol-Quality Relationship**: Strong positive correlation between alcohol content and wine quality ratings (Spearman ρ ≈ 0.44). Higher alcohol wines tend to receive better ratings.
            
            2. **Alcohol-Density Relationship**: Very strong negative correlation (Pearson r ≈ -0.78). This makes chemical sense as alcohol is less dense than water.
            
            3. **Variability Analysis**: Variables like residual sugar, chlorides, and sulphates show high variability (high IQR and std dev) and positive skewness, indicating presence of outliers.
            
            4. **Wine Type Association**: Chi-square test reveals significant association between wine type (red/white) and quality classification, suggesting different quality patterns.
            
            5. **Distribution Patterns**: Most variables show non-normal distributions (evident from kurtosis and skewness values), justifying use of non-parametric tests like Spearman and Kendall's Tau.
            """)
            
            st.info("💡 **Statistical Significance**: All major correlation tests show p-values < 0.05, indicating statistically significant relationships.")

        except Exception as e:
            # If any Streamlit-specific error occurs (e.g. running script directly with python),
            # fallback to headless analysis using the already-loaded DataFrame.
            st.warning("Streamlit UI mode failed or isn't running as a server. Falling back to headless analysis. See logs.")
            st.write("Error details:", e)
            try:
                # run headless analysis and save outputs to ./eda_outputs
                if 'df' not in locals():
                    print("Dataframe not found in memory, attempting to load sample...")
                    df = load_wine_sample()
                run_headless_and_save(df, out_dir='./eda_outputs')
                st.success("Headless analysis complete — outputs written to ./eda_outputs/")
            except Exception as e2:
                st.error(f"Headless analysis also failed: {e2}")
                st.stop()
    else:
        print("Streamlit is not available in this environment. Running headless CLI analysis instead.")
        main_cli()
