"""
Well Log Prediction - Streamlit App
===================================
Interactive version of the multi-well sonic prediction pipeline. Predicts
either compressional sonic (DT) or shear sonic (DTS); for shear, DT can also
be used as an input.
  * Upload LAS files (or use built-in synthetic demo wells)
  * Mnemonic standardization across service-company naming conventions
  * Physical-range cleaning with adjustable limits
  * Blind-well validation (whole wells held out via GroupShuffleSplit)
  * Linear Regression / Random Forest / XGBoost / Keras MLP comparison, plus a
    "Compare runs" table (e.g. DTS with vs without DT as an input)
  * Log-track plot, metrics table, and CSV / LAS download of predictions
  * FINAL MODEL: retrain on every well with a measured DT, then predict DT for
    ALL wells (including wells with a sonic gap or no sonic log at all) and
    export the results as CSV, LAS, or a ZIP of LAS files
  * Export the CLEANED logs (CSV / LAS) straight from the Data tab, before
    any training
  * SAVE the final model to a file, then use "Apply saved model" mode to load
    it later and predict DT for new wells without retraining

Run with:
    pip install streamlit lasio numpy pandas matplotlib scikit-learn xgboost tensorflow
    streamlit run well_log_ml_app.py

XGBoost and TensorFlow are optional: if either is not installed, that model
is simply hidden from the sidebar instead of crashing the app.
"""

import io
import json
import os
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# scikit-learn is required for the ML pipeline, but we keep the import guarded
# so the app still starts with a friendly message if the environment is missing
# the package (common in VS Code / Poetry / Conda setups).
try:
    from sklearn.model_selection import GroupShuffleSplit  # type: ignore[reportMissingImports]
    from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingImports]
    from sklearn.ensemble import RandomForestRegressor  # type: ignore[reportMissingImports]
    from sklearn.linear_model import LinearRegression  # type: ignore[reportMissingImports]
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore[reportMissingImports]
    HAS_SKLEARN = True
except ImportError:
    GroupShuffleSplit = None
    StandardScaler = None
    RandomForestRegressor = None
    LinearRegression = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    HAS_SKLEARN = False

# joblib ships with scikit-learn; used to save / load trained models
try:
    import joblib  # type: ignore[reportMissingImports]
    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False

# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------------
# Wrapped in try/except so the app still starts if a heavy library is missing.
try:
    import lasio  # type: ignore[reportMissingImports]
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False

try:
    import xgboost as xgb  # type: ignore[reportMissingImports]
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Standard name -> alternative mnemonics in PRIORITY ORDER (first match wins)
LAS_ALIASES = {
    'GR':   ['GR', 'SGR', 'GRD', 'CGR', 'GAM'],
    'RHOB': ['RHOB', 'DEN', 'ZDEN', 'RHOZ', 'BD'],
    'NPHI': ['NPHI', 'CNC', 'NPHI_LS', 'PHIN', 'HNPHI'],
    'ILD':  ['ILD', 'LLD', 'RT', 'AT90', 'AHT90', 'RDEP','RES'],
    'CALI': ['CALI', 'CAL', 'CALS', 'HCAL'],
    'DT':   ['DT', 'DTCO', 'AC', 'SONI', 'DTP'],
    'DTS':  ['DTS', 'DTSM', 'DTSH', 'DTSD', 'DT_SHEAR', 'SDT'],
}

# Logs the app can predict. key -> label shown in the UI
TARGETS = {'DT': 'DT (compressional sonic)', 'DTS': 'DTS (shear sonic)'}
DEPTH_ALIASES = ['DEPT', 'DEPTH', 'MD']

st.set_page_config(page_title="Well Log ML", page_icon="🛢️", layout="wide")


# ---------------------------------------------------------------------------
# 1. DATA LOADING & STANDARDIZATION
# ---------------------------------------------------------------------------
def standardize_curve_names(df, alias_dict):
    """Rename columns to standard mnemonics (one raw column per standard name)."""
    df.columns = [str(c).upper() for c in df.columns]
    rename_map = {}

    # Depth first
    for alias in DEPTH_ALIASES:
        if alias in df.columns:
            rename_map[alias] = 'DEPTH'
            break

    for standard, aliases in alias_dict.items():
        for alias in aliases:
            if alias.upper() in df.columns:
                rename_map[alias.upper()] = standard
                break  # highest-priority alias wins

    df = df.rename(columns=rename_map)
    # Safety net against duplicate column names
    return df.loc[:, ~df.columns.duplicated()]


@st.cache_data(show_spinner="Reading LAS files...")
def load_uploaded_las(file_payload):
    """Parse uploaded LAS files.

    `file_payload` is a tuple of (filename, bytes) pairs. Using plain tuples
    (rather than Streamlit UploadedFile objects) makes the input hashable, so
    st.cache_data can skip re-parsing when nothing has changed.
    """
    wells, errors = [], []
    for name, raw in file_payload:
        try:
            # lasio reads from a text stream; decode the uploaded bytes first
            las = lasio.read(io.StringIO(raw.decode('utf-8', errors='ignore')))
            df_well = las.df().reset_index()
            df_well = standardize_curve_names(df_well, LAS_ALIASES)
            well_name = las.well.WELL.value if 'WELL' in las.well else name
            df_well['WELL_NAME'] = str(well_name)
            wells.append(df_well)
        except Exception as e:  # one bad file shouldn't kill the batch
            errors.append(f"{name}: {e}")

    if not wells:
        return None, errors
    return pd.concat(wells, axis=0, ignore_index=True), errors


@st.cache_data
def simulate_demo_data():
    """4 synthetic wells, each using different curve mnemonics.

    DT is built from the other logs plus noise so models have something real
    to learn. Metrics on this data do not reflect real-world performance.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    variations = [
        # DTS: shear mnemonic, or None = this well has no shear log
        {'GR': 'SGR', 'RHOB': 'ZDEN', 'ILD': 'RT',   'DT': 'DTCO', 'DTS': 'DTSM', 'name': 'WELL_A_CHEVRON'},
        {'GR': 'GR',  'RHOB': 'DEN',  'ILD': 'AT90', 'DT': 'AC',   'DTS': 'DTS',  'name': 'WELL_B_SHELL'},
        {'GR': 'CGR', 'RHOB': 'RHOB', 'ILD': 'RDEP', 'DT': 'DT',   'DTS': 'DTSH', 'name': 'WELL_C_TOTAL'},
        {'GR': 'GAM', 'RHOB': 'RHOZ', 'ILD': 'ILD',  'DT': 'SONI', 'DTS': None,   'name': 'WELL_D_BP'},
    ]
    frames = []
    for var in variations:
        n = 1000
        gr = rng.uniform(10, 130, n)
        rhob = rng.uniform(2.1, 2.7, n)
        nphi = rng.uniform(0.05, 0.38, n)
        ild = 10 ** rng.uniform(0.3, 2.5, n)
        dt = 160 - 45 * rhob + 80 * nphi + 0.05 * gr + rng.normal(0, 2.0, n)

        df_m = pd.DataFrame({
            'DEPTH': np.linspace(3000, 4000, n),
            'CALI': rng.uniform(8, 12, n),
            var['GR']: gr,
            var['RHOB']: rhob,
            'NPHI': nphi,
            var['ILD']: ild,
            var['DT']: dt,
        })
        if var['DTS']:
            # Vp/Vs rises with shale (GR) and porosity; plus noise
            vpvs = 1.65 + 0.003 * (gr - 70) + 0.8 * (nphi - 0.2)
            df_m[var['DTS']] = dt * vpvs + rng.normal(0, 4.0, n)
        df_m = standardize_curve_names(df_m, LAS_ALIASES)
        df_m['WELL_NAME'] = var['name']
        frames.append(df_m)

    # A 5th well with NO sonic log: it is skipped by training / blind
    # validation, but the final model (All wells tab) can predict its DT.
    n = 1000
    frames.append(pd.DataFrame({
        'DEPTH': np.linspace(2500, 3500, n),
        'CALI': rng.uniform(8, 12, n),
        'GR': rng.uniform(10, 130, n),
        'RHOB': rng.uniform(2.1, 2.7, n),
        'NPHI': rng.uniform(0.05, 0.38, n),
        'ILD': 10 ** rng.uniform(0.3, 2.5, n),
        'DT': np.nan,
        'WELL_NAME': 'WELL_E_NO_SONIC',
    }))
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------------
# Every log that gets outlier cleaning (DT included, it is the target)
CLEAN_CURVES = ['GR', 'RHOB', 'NPHI', 'ILD', 'CALI', 'DT', 'DTS']
# Curves whose spikes are judged on a log10 scale (resistivity spans decades)
LOG_SCALE_CURVES = {'ILD'}


def clean_data(df, feature_curves, limits, spike_pct=0.0, require_target=True,
               return_report=False, target='DT'):
    """Remove outliers from ALL logs, drop unusable rows, add log10 resistivity.

    Cleaning works value by value, in two steps:
      1. Physical range: any reading outside `limits[curve] = (min, max)` is
         set to NaN. This applies to every curve in CLEAN_CURVES that exists,
         whether or not it is a model input.
      2. Spikes (optional, `spike_pct` > 0): within EACH well, readings below
         the `spike_pct` percentile or above the 100 - `spike_pct` percentile
         are set to NaN. Done per well so one well's range can't clip another.

    Then rows are dropped only if a curve the model actually NEEDS is missing:
    the selected input curves, plus the `target` log (DT or DTS) when
    `require_target=True`. So a bad caliper reading won't throw away a row
    unless CALI is a selected input, and a bad target reading turns that depth
    into a "predict it here" row.

    `require_target=True`  -> keep only rows with a valid measured target
                              (training).
    `require_target=False` -> also keep rows where the target is missing or was
                              removed as an outlier, as long as the inputs are
                              valid (prediction).

    The flagging doesn't depend on `require_target`, so the training rows are
    always a subset of the prediction rows.

    With `return_report=True`, also returns a table of how many values each
    step removed per curve.
    """
    cleaned = df.copy()
    report = []

    # Rows with no depth / well name are unusable
    cleaned = cleaned.dropna(subset=[c for c in ['DEPTH', 'WELL_NAME'] if c in cleaned.columns])

    curves = [c for c in CLEAN_CURVES if c in cleaned.columns]
    for c in curves:
        cleaned[c] = pd.to_numeric(cleaned[c], errors='coerce')
        n_before = int(cleaned[c].notna().sum())

        # Step 1: physical range (skipped for a curve with no limits, e.g. a
        # model saved before that curve was added)
        if c in limits:
            lo, hi = limits[c]
            out_of_range = cleaned[c].notna() & ~cleaned[c].between(lo, hi)
        else:
            out_of_range = pd.Series(False, index=cleaned.index)
        cleaned.loc[out_of_range, c] = np.nan

        # Step 2: per-well percentile spikes
        n_spikes = 0
        if spike_pct > 0:
            vals = cleaned[c]
            if c in LOG_SCALE_CURVES:
                vals = np.log10(vals.where(vals > 0))
            grp = vals.groupby(cleaned['WELL_NAME'])
            p_lo = grp.transform(lambda s: s.quantile(spike_pct / 100))
            p_hi = grp.transform(lambda s: s.quantile(1 - spike_pct / 100))
            spikes = vals.notna() & ((vals < p_lo) | (vals > p_hi))
            cleaned.loc[spikes, c] = np.nan
            n_spikes = int(spikes.sum())

        report.append({'Curve': c,
                       'Valid readings (raw)': n_before,
                       'Outside range': int(out_of_range.sum()),
                       'Spikes removed': n_spikes,
                       'Valid after cleaning': int(cleaned[c].notna().sum())})

    # Keep only rows that have every curve the model needs
    needed = feature_curves + ([target] if require_target else [])
    cleaned = cleaned.dropna(subset=[c for c in needed if c in cleaned.columns])

    if 'ILD' in cleaned.columns:
        cleaned['ILD_log10'] = np.log10(cleaned['ILD'])  # NaN where ILD is missing

    if return_report:
        return cleaned, pd.DataFrame(report).set_index('Curve')
    return cleaned


# ---------------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------------
def fit_models(X_train_s, y_train, selected_models, params):
    """Fit every selected model on already-scaled training data.

    Shared by the blind-well validation run AND the final all-wells model, so
    both use exactly the same model settings.

    Returns:
        fitted  - {prediction column: (label, predict_fn, model)}. predict_fn
                  takes a scaled feature matrix and returns a 1-D numpy array;
                  model is the fitted object (kept so it can be saved).
        history - Keras training history (dict), or None if the MLP wasn't used.
    """
    fitted, history = {}, None

    if 'Linear Regression' in selected_models:
        # Simple baseline. Unlike tree models it can extrapolate, which helps
        # for near-linear relations such as DTS from DT (Castagna-style).
        lr = LinearRegression()
        lr.fit(X_train_s, y_train)
        fitted['PRED_LR'] = ('Linear Regression', lr.predict, lr)

    if 'Random Forest' in selected_models:
        rf = RandomForestRegressor(
            n_estimators=params['rf_trees'], max_depth=params['rf_depth'],
            random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        fitted['PRED_RF'] = ('Random Forest', rf.predict, rf)

    if 'XGBoost' in selected_models:
        xg = xgb.XGBRegressor(
            n_estimators=params['xgb_trees'], learning_rate=params['xgb_lr'],
            max_depth=params['xgb_depth'], random_state=RANDOM_STATE)
        xg.fit(X_train_s, y_train)
        fitted['PRED_XGB'] = ('XGBoost', xg.predict, xg)

    if 'Deep Learning (MLP)' in selected_models:
        tf.random.set_seed(RANDOM_STATE)
        mlp = Sequential([
            Input(shape=(X_train_s.shape[1],)),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(1),
        ])
        mlp.compile(optimizer='adam', loss='mse')
        stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # NOTE: validation_split uses the LAST 15% of the rows as given.
        hist = mlp.fit(X_train_s, y_train, validation_split=0.15,
                       epochs=params['mlp_epochs'], batch_size=32,
                       callbacks=[stop], verbose=0)
        history = hist.history
        fitted['PRED_DL'] = ('Deep Learning (MLP)',
                             lambda X, m=mlp: m.predict(X, verbose=0).flatten(),
                             mlp)

    return fitted, history


def train_and_predict(df_clean, features, selected_models, test_size, params, target='DT'):
    """Split by well, scale, train every selected model, return results."""
    X = df_clean[features]
    y = df_clean[target]
    groups = df_clean['WELL_NAME']

    # Hold out ENTIRE wells so neighbouring depth samples can't leak
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    blind_df = df_clean.iloc[test_idx].copy()
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit scaler on TRAIN only to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    fitted, history = fit_models(X_train_s, y_train, selected_models, params)

    pred_cols = {}  # model label -> prediction column name
    for col, (label, predict, _model) in fitted.items():
        blind_df[col] = predict(X_test_s)
        pred_cols[label] = col

    # Quantitative comparison on the blind wells
    rows = []
    for label, col in pred_cols.items():
        rows.append({
            'Model': label,
            'RMSE (us/ft)': np.sqrt(mean_squared_error(y_test, blind_df[col])),
            'MAE (us/ft)': mean_absolute_error(y_test, blind_df[col]),
            'R²': r2_score(y_test, blind_df[col]),
        })
    metrics = pd.DataFrame(rows).set_index('Model')

    return {
        'blind_df': blind_df,
        'pred_cols': pred_cols,
        'metrics': metrics,
        'train_wells': sorted(df_clean.iloc[train_idx]['WELL_NAME'].unique()),
        'test_wells': sorted(blind_df['WELL_NAME'].unique()),
        'history': history,
        'target': target,
        'features': features,
    }


def train_final_and_predict_all(df_train, df_predict, features, selected_models, params,
                                target='DT'):
    """FINAL MODEL: train on every well with a measured target (no hold-out),
    then predict the target (DT or DTS) for every row in `df_predict`.

    df_train   - cleaned rows WITH a measured target (the training set)
    df_predict - cleaned rows with valid input curves, target optional. It is a
                 superset of df_train, so it also contains wells that have a
                 gap in the target log or no target log at all.

    Because the training wells are part of the prediction set, predictions on
    them are IN-SAMPLE and look better than they really are. Use the blind-well
    validation (Results tab) to judge accuracy; use this for filling in logs.
    """
    X_train = df_train[features]
    y_train = df_train[target].to_numpy()

    # Scaler fitted on ALL training rows; reused for every well we predict
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Shuffle rows so the MLP's validation_split (last 15%) isn't just the
    # tail of the last well. Harmless for the tree models.
    perm = np.random.RandomState(RANDOM_STATE).permutation(len(X_train_s))
    fitted, history = fit_models(X_train_s[perm], y_train[perm], selected_models, params)

    out = df_predict.copy()
    X_all_s = scaler.transform(out[features])
    pred_cols = {}
    for col, (label, predict, _model) in fitted.items():
        out[col] = predict(X_all_s)
        pred_cols[label] = col

    # Flag which rows the model actually saw during training
    out['USED_IN_TRAINING'] = out.index.isin(df_train.index)

    # In-sample fit on the training rows (optimistic by construction)
    seen = out[out['USED_IN_TRAINING']]
    rows = []
    for label, col in pred_cols.items():
        rows.append({
            'Model': label,
            'RMSE (us/ft)': np.sqrt(mean_squared_error(seen[target], seen[col])),
            'R²': r2_score(seen[target], seen[col]),
        })
    in_sample = pd.DataFrame(rows).set_index('Model')

    # One line per well: how much was measured vs predicted-only
    summary = (out.groupby('WELL_NAME')
                  .agg(rows=('DEPTH', 'size'),
                       measured=('USED_IN_TRAINING', 'sum'),
                       top=('DEPTH', 'min'),
                       base=('DEPTH', 'max'))
                  .reset_index())
    summary['predicted_only'] = summary['rows'] - summary['measured']
    summary.columns = ['Well', 'Rows', f'Rows with measured {target}',
                       'Top depth', 'Base depth', 'Rows predicted only']

    return {
        'predictions': out,
        'pred_cols': pred_cols,
        'in_sample': in_sample,
        'summary': summary,
        'features': features,
        'history': history,
        'target': target,
        # Kept so the model can be saved and re-applied later
        'scaler': scaler,
        'models': {col: model for col, (_label, _predict, model) in fitted.items()},
    }


# ---------------------------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------------------------
PRED_STYLES = {
    'PRED_LR':  dict(color='gray',       linestyle='-',  label='Linear Regression', alpha=0.8),
    'PRED_RF':  dict(color='purple',     linestyle='-.', label='Random Forest'),
    'PRED_XGB': dict(color='teal',       linestyle='--', label='XGBoost'),
    'PRED_DL':  dict(color='darkorange', linestyle=':',  label='Deep Learning'),
}


def plot_blind_well(blind_df, well_name, pred_cols, title=None, target='DT'):
    """GR | RHOB | [DT] | target (actual vs predicted) tracks for one well.

    When the target is shear (DTS), an extra track shows the compressional DT
    so the two sonics can be compared. Used for blind wells, final-model wells
    and saved-model predictions (pass `title` to relabel).
    """
    well_df = blind_df[blind_df['WELL_NAME'] == well_name].sort_values('DEPTH')
    show_dt = target != 'DT' and 'DT' in well_df.columns and well_df['DT'].notna().any()
    n_tracks = 4 if show_dt else 3

    fig, ax = plt.subplots(1, n_tracks, figsize=(3.3 * n_tracks, 8), sharey=True)
    fig.suptitle(title or f"Blind Prediction Profile: {well_name}",
                 fontsize=14, weight='bold')

    # Track 1: Gamma Ray (only if used as a feature / present)
    if 'GR' in well_df.columns:
        ax[0].plot(well_df['GR'], well_df['DEPTH'], color='green')
    ax[0].set_xlabel('GR (API)')
    ax[0].set_ylabel('Depth')
    ax[0].invert_yaxis()  # depth increases downward; sharey flips all tracks

    # Track 2: Density
    if 'RHOB' in well_df.columns:
        ax[1].plot(well_df['RHOB'], well_df['DEPTH'], color='red')
    ax[1].set_xlabel('RHOB (g/cc)')

    # Optional track: compressional sonic, when predicting shear
    if show_dt:
        ax[2].plot(well_df['DT'], well_df['DEPTH'], color='royalblue')
        ax[2].set_xlabel('DT (us/ft)')

    # Last track: actual target vs each model's prediction
    # Wells with no measured target have nothing to draw for "actual"
    tax = ax[-1]
    if target in well_df.columns and well_df[target].notna().any():
        tax.plot(well_df[target], well_df['DEPTH'], color='black', label=f'Actual {target}')
    for col in pred_cols.values():
        tax.plot(well_df[col], well_df['DEPTH'], **PRED_STYLES[col])
    tax.set_xlabel(f"{TARGETS.get(target, target).split(' (')[0]} "
                   f"{'shear' if target == 'DTS' else 'sonic'} (us/ft)")
    tax.legend(loc='upper right')

    for a in ax:
        a.grid(True, linestyle=':')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4b. LAS EXPORT
# ---------------------------------------------------------------------------
# Units / descriptions written into the LAS curve headers
CURVE_META = {
    'GR':       ('GAPI',  'Gamma ray'),
    'RHOB':     ('G/CC',  'Bulk density'),
    'NPHI':     ('V/V',   'Neutron porosity'),
    'ILD':      ('OHMM',  'Deep resistivity'),
    'CALI':     ('IN',    'Caliper'),
    'DT':       ('US/F',  'Compressional sonic slowness (measured)'),
    'DTS':      ('US/F',  'Shear sonic slowness (measured)'),
}
# Predicted curves are exported as <target>_<model>, e.g. DT_RF or DTS_XGB
MODEL_SUFFIX = {'PRED_LR': 'LR', 'PRED_RF': 'RF', 'PRED_XGB': 'XGB', 'PRED_DL': 'MLP'}
MODEL_NAME = {'PRED_LR': 'Linear Regression', 'PRED_RF': 'Random Forest',
              'PRED_XGB': 'XGBoost', 'PRED_DL': 'MLP'}
NULL_VALUE = -999.25


def export_name(col, target):
    """Output mnemonic for a column: PRED_RF -> DT_RF / DTS_RF, others as is."""
    return f"{target}_{MODEL_SUFFIX[col]}" if col in MODEL_SUFFIX else col


def curve_meta(col, target):
    """(unit, description) for a curve in the LAS header."""
    if col in MODEL_SUFFIX:
        kind = 'Shear' if target == 'DTS' else 'Compressional'
        return 'US/F', f"{kind} sonic slowness predicted by {MODEL_NAME[col]}"
    return CURVE_META.get(col, ('', col))


def for_export(df, target):
    """Copy of a predictions table with PRED_* columns renamed for CSV export."""
    return df.rename(columns={c: export_name(c, target) for c in MODEL_SUFFIX})


def safe_filename(name):
    """Make a well name safe to use as a file name."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def build_las_text(well_df, well_name, depth_unit, target='DT'):
    """Return LAS 2.0 text for one well: depth, input logs, measured logs
    (where they exist) and every model's prediction of `target`, named e.g.
    DT_RF or DTS_XGB.

    Only rows that survived cleaning are exported, so the depth spacing may
    have gaps where bad data was removed. Missing values are written as the
    LAS null value (-999.25).
    """
    las = lasio.LASFile()
    las.well['WELL'] = lasio.HeaderItem('WELL', value=str(well_name), descr='Well name')
    las.well['NULL'] = lasio.HeaderItem('NULL', value=NULL_VALUE, descr='Null value')

    # Sort by depth and drop curves that are completely empty for this well
    # (e.g. DT in a well that has no sonic log).
    well_df = well_df.sort_values('DEPTH').dropna(axis=1, how='all')

    # lasio renamed add_curve() -> append_curve() in newer releases; use
    # whichever this installed version provides.
    add = getattr(las, 'append_curve', None) or las.add_curve

    # The first curve in a LAS file is the depth index
    add('DEPT', well_df['DEPTH'].to_numpy(dtype=float), unit=depth_unit, descr='Depth')

    # Everything except depth, well name, and internal helper columns
    skip = {'DEPTH', 'WELL_NAME', 'ILD_log10', 'USED_IN_TRAINING'}
    for col in well_df.columns:
        if col in skip:
            continue
        # Skip non-numeric columns (e.g. text curves from uploaded LAS files)
        values = pd.to_numeric(well_df[col], errors='coerce').to_numpy(dtype=float)
        if np.isnan(values).all():
            continue
        unit, descr = curve_meta(col, target)
        # Write NaN explicitly as the null value rather than relying on lasio
        values = np.where(np.isnan(values), NULL_VALUE, values)
        add(export_name(col, target), values, unit=unit, descr=descr)

    buf = io.StringIO()
    las.write(buf, version=2.0)
    return buf.getvalue()


@st.cache_data(show_spinner="Building LAS files...")
def build_las_zip(df, depth_unit, suffix="_predicted", target='DT'):
    """ZIP containing one LAS file per well in `df` (cached so Streamlit
    reruns don't rebuild it unless the data or depth unit changes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for well, g in df.groupby('WELL_NAME'):
            zf.writestr(f"{safe_filename(well)}{suffix}.las",
                        build_las_text(g, well, depth_unit, target))
    return buf.getvalue()


def cleaned_logs_for_export(df_raw, limits, spike_pct, include_other):
    """All logs after outlier cleaning, for export BEFORE any training.

    Unlike the training data, no rows are dropped for a missing input or
    target: every depth is kept, bad readings are simply written as null. Rows
    where every log is null are removed.

    include_other=False -> depth + the standard curves (GR, RHOB, NPHI, ILD,
                           CALI, DT, DTS) only.
    include_other=True  -> also any other numeric curves from the LAS files
                           (e.g. PEF, DRHO). These are NOT outlier-cleaned.
    """
    cleaned = clean_data(df_raw, [], limits, spike_pct, require_target=False)
    std = [c for c in CLEAN_CURVES if c in cleaned.columns]
    if include_other:
        other = [c for c in cleaned.columns
                 if c not in std and c not in ('DEPTH', 'WELL_NAME', 'ILD_log10')
                 and pd.api.types.is_numeric_dtype(cleaned[c])]
    else:
        other = []
    out = cleaned[['WELL_NAME', 'DEPTH'] + std + other]
    return out.dropna(subset=std + other, how='all')


# ---------------------------------------------------------------------------
# 4c. SAVE / LOAD TRAINED MODELS
# ---------------------------------------------------------------------------
# A saved model is a .zip file holding:
#   meta.json       - inputs, cleaning limits, training wells, scores, versions
#   sklearn.joblib  - the scaler + Random Forest / XGBoost models
#   mlp.keras       - the Keras MLP (only if it was trained)
BUNDLE_VERSION = 1


def _versions():
    import sklearn  # type: ignore[reportMissingImports]
    v = {'scikit-learn': sklearn.__version__, 'numpy': np.__version__}
    if HAS_XGB:
        v['xgboost'] = xgb.__version__
    if HAS_TF:
        v['tensorflow'] = tf.__version__
    return v


def make_model_bundle(final, meta):
    """Serialise the final model (scaler + every trained model) to zip bytes."""
    tree_models = {c: m for c, m in final['models'].items() if c != 'PRED_DL'}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('meta.json', json.dumps(meta, indent=2, default=float))

        sk_buf = io.BytesIO()
        joblib.dump({'scaler': final['scaler'], 'models': tree_models}, sk_buf, compress=3)
        zf.writestr('sklearn.joblib', sk_buf.getvalue())

        if 'PRED_DL' in final['models']:
            # Keras saves to a file path, so go through a temp folder
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'mlp.keras')
                final['models']['PRED_DL'].save(path)
                with open(path, 'rb') as fh:
                    zf.writestr('mlp.keras', fh.read())
    return buf.getvalue()


@st.cache_resource(show_spinner="Loading model...")
def load_model_bundle(raw):
    """Inverse of make_model_bundle. Returns dict(meta, scaler, models, warnings)."""
    warnings_ = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if 'meta.json' not in names or 'sklearn.joblib' not in names:
            raise ValueError("This zip is not a model saved by this app "
                             "(meta.json / sklearn.joblib missing).")
        meta = json.loads(zf.read('meta.json'))
        sk = joblib.load(io.BytesIO(zf.read('sklearn.joblib')))
        models = dict(sk['models'])

        if 'mlp.keras' in names:
            if HAS_TF:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, 'mlp.keras')
                    with open(path, 'wb') as fh:
                        fh.write(zf.read('mlp.keras'))
                    models['PRED_DL'] = tf.keras.models.load_model(path)
            else:
                warnings_.append("This file contains an MLP model, but TensorFlow is not "
                                 "installed here, so the MLP is unavailable.")

    if 'XGBoost' in meta.get('models', {}).values() and not HAS_XGB:
        warnings_.append("XGBoost is not installed, so the XGBoost model may not load.")

    # Different library versions usually still work, but say so
    try:
        now = _versions()
        for lib, ver in meta.get('versions', {}).items():
            if lib in now and now[lib] != ver:
                warnings_.append(f"Model saved with {lib} {ver}; this app is running "
                                 f"{now[lib]}. Predictions should still work, but check them.")
    except Exception:
        pass

    return {'meta': meta, 'scaler': sk['scaler'], 'models': models, 'warnings': warnings_}


def predict_with_bundle(bundle, df, cols):
    """Add a prediction column for each model in `cols` to a copy of `df`."""
    X = bundle['scaler'].transform(df[bundle['meta']['model_features']])
    out = df.copy()
    for col in cols:
        model = bundle['models'][col]
        if col == 'PRED_DL':
            out[col] = model.predict(X, verbose=0).flatten()
        else:
            out[col] = np.asarray(model.predict(X)).ravel()
    return out


def score_table(df, pred_cols, target='DT'):
    """RMSE / MAE / R² for rows that have a measured target (empty if none)."""
    meas = df[df[target].notna()]
    rows = []
    if len(meas) > 1:
        for label, col in pred_cols.items():
            rows.append({'Model': label,
                         'RMSE (us/ft)': np.sqrt(mean_squared_error(meas[target], meas[col])),
                         'MAE (us/ft)': mean_absolute_error(meas[target], meas[col]),
                         'R²': r2_score(meas[target], meas[col])})
    return pd.DataFrame(rows).set_index('Model') if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4d. "APPLY SAVED MODEL" MODE
# ---------------------------------------------------------------------------
def render_apply_mode():
    """Load a saved model + new LAS files, predict its target (DT or DTS),
    show and export results."""
    st.header("🔮 Apply a saved model to new wells")
    st.caption("Load a model saved from the **All wells (final model)** tab of Train mode, "
               "then predict DT or DTS (whichever the model was trained for) for any wells "
               "that have the same input curves. No retraining happens here.")

    with st.sidebar:
        st.header("1. Saved model")
        model_file = st.file_uploader("Model file (.zip)", type=["zip"], key="apply_model")
        st.caption("⚠️ Only load model files you made or trust: they contain pickled "
                   "Python objects, which can run code when opened.")
        st.header("2. Wells to predict")
        src = st.radio("Wells", ["Upload LAS files", "Demo (synthetic wells)"], key="apply_src")
        las_files = []
        if src == "Upload LAS files":
            if not HAS_LASIO:
                st.error("`lasio` is not installed. Run `pip install lasio`.")
            las_files = st.file_uploader("LAS files", type=["las", "LAS"],
                                         accept_multiple_files=True, key="apply_las")

    if not HAS_JOBLIB:
        st.error("`joblib` is not installed (it normally comes with scikit-learn).")
        return
    if model_file is None:
        st.info("Upload a saved model file in the sidebar.")
        return

    try:
        bundle = load_model_bundle(model_file.getvalue())
    except Exception as e:
        st.error("Could not read this model file.")
        st.exception(e)
        return
    meta = bundle['meta']
    tgt = meta.get('target', 'DT')  # models saved before DTS support predict DT
    for w in bundle['warnings']:
        st.warning(w)

    # Model summary ------------------------------------------------------------
    st.subheader("Model")
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Predicts", TARGETS.get(tgt, tgt))
    c1.metric("Input curves", ", ".join(meta['input_curves']))
    c2.metric("Training wells", len(meta['train_wells']))
    c3.metric("Saved", meta.get('created', '?').replace('T', ' '))
    if meta.get('blind_validation'):
        st.caption("Blind-well validation recorded when this model was trained:")
        st.dataframe(pd.DataFrame(meta['blind_validation']).T.round(3),
                     use_container_width=True)
    with st.expander("All model details"):
        st.json(meta)

    labels = {col: lab for col, lab in meta['models'].items() if col in bundle['models']}
    chosen_labels = st.multiselect("Models to apply", list(labels.values()),
                                   default=list(labels.values()))
    chosen = [col for col, lab in labels.items() if lab in chosen_labels]
    if not chosen:
        st.warning("Select at least one model.")
        return

    # Wells --------------------------------------------------------------------
    df_new, errs = None, []
    if src == "Demo (synthetic wells)":
        df_new = simulate_demo_data()
    elif las_files and HAS_LASIO:
        df_new, errs = load_uploaded_las(tuple((f.name, f.getvalue()) for f in las_files))
    for err in errs:
        st.warning(f"Could not read {err}")
    if df_new is None:
        st.info("Upload the LAS files you want to predict in the sidebar.")
        return

    req = meta['input_curves']

    # A shear model that uses DT as an input can borrow a PREDICTED DT where
    # the measured one is missing, e.g. from LAS files exported after applying
    # a DT model (they contain DT_RF / DT_XGB / DT_MLP curves).
    if 'DT' in req:
        pred_dt_cols = [c for c in ('DT_LR', 'DT_RF', 'DT_XGB', 'DT_MLP') if c in df_new.columns]
        if pred_dt_cols:
            fill_from = st.selectbox(
                "Where measured DT is missing, use predicted DT from",
                ["Don't fill (skip those depths)"] + pred_dt_cols,
                help="These wells contain predicted DT curves (from a DT model). Using one "
                     "lets this shear model also predict where no DT was logged. The "
                     "DT_FROM_PRED column marks those depths (1 = predicted DT used).")
            if fill_from in pred_dt_cols:
                if 'DT' not in df_new.columns:
                    df_new = df_new.assign(DT=np.nan)
                use_pred = df_new['DT'].isna() & df_new[fill_from].notna()
                df_new = df_new.assign(DT=df_new['DT'].fillna(df_new[fill_from]),
                                       DT_FROM_PRED=use_pred.astype(float))
                st.caption(f"Filled DT from {fill_from} at {int(use_pred.sum()):,} depths.")
        elif 'DT' not in df_new.columns or df_new['DT'].isna().all():
            st.info("This model uses DT as an input. Wells without DT can't be predicted. "
                    "Tip: apply a DT model to them first and export the LAS files; loading "
                    "those here lets you fill the missing DT with the predicted one.")

    missing = [c for c in req if c not in df_new.columns]
    if missing:
        st.error(f"These wells don't have curve(s) the model needs: {', '.join(missing)}. "
                 "Check the mnemonics, or add them to `LAS_ALIASES`.")
        return
    if tgt not in df_new.columns:
        df_new = df_new.assign(**{tgt: np.nan})

    # Same cleaning as when the model was trained
    limits_saved = {k: tuple(v) for k, v in meta['limits'].items()}
    df_pred, report = clean_data(df_new, req, limits_saved, meta.get('spike_pct', 0.0),
                                 require_target=False, return_report=True, target=tgt)
    with st.expander("Cleaning applied (same limits as when the model was trained)"):
        st.dataframe(pd.DataFrame(meta['limits'], index=['min', 'max']).T,
                     use_container_width=True)
        st.dataframe(report, use_container_width=True)

    skipped = sorted(set(df_new['WELL_NAME']) - set(df_pred['WELL_NAME']))
    if skipped:
        st.warning("No usable rows (inputs missing or out of range) in: " + ", ".join(skipped))
    if df_pred.empty:
        st.error("No rows left to predict after cleaning.")
        return

    # Rows outside the range the model saw in training -> extrapolation
    ranges = meta.get('feature_ranges', {})
    rows_out = []
    for f, (lo, hi) in ranges.items():
        if f in df_pred.columns:
            pct = 100 * (~df_pred[f].between(lo, hi)).mean()
            if pct >= 5:
                rows_out.append(f"{f.replace('ILD_log10', 'ILD')} ({pct:.0f}% of rows)")
    if rows_out:
        st.warning("Some inputs are outside the range the model was trained on (1st-99th "
                   "percentile), so predictions there are extrapolations: " + ", ".join(rows_out))

    overlap = sorted(set(df_pred['WELL_NAME']) & set(meta['train_wells']))
    if overlap:
        st.info("These wells were in the model's training data, so their scores are "
                "in-sample (optimistic): " + ", ".join(overlap))

    sig = (model_file.name, model_file.size, src,
           tuple(sorted((f.name, f.size) for f in las_files)) if las_files else (),
           tuple(chosen), len(df_pred))
    if st.button(f"Predict {tgt}", type="primary"):
        with st.spinner("Predicting..."):
            out = predict_with_bundle(bundle, df_pred, chosen)
            st.session_state['applied'] = {'sig': sig, 'out': out,
                                           'pred_cols': {labels[c]: c for c in chosen}}

    applied = st.session_state.get('applied')
    if applied is None or applied['sig'] != sig:
        st.info(f"Click **Predict {tgt}** to run the model on these wells.")
        return
    out, pred_cols = applied['out'], applied['pred_cols']

    # Results ------------------------------------------------------------------
    st.subheader("Predictions per well")
    summary = (out.groupby('WELL_NAME')
                  .agg(Rows=('DEPTH', 'size'), **{f'Rows with measured {tgt}': (tgt, 'count')},
                       **{'Top depth': ('DEPTH', 'min'), 'Base depth': ('DEPTH', 'max')})
                  .reset_index().rename(columns={'WELL_NAME': 'Well'}))
    st.dataframe(summary, use_container_width=True, hide_index=True)

    scores = score_table(out, pred_cols, tgt)
    if not scores.empty:
        st.subheader(f"Accuracy where these wells have a measured {tgt}")
        st.dataframe(scores.round(3), use_container_width=True)
        if not overlap:
            st.caption("None of these wells were used in training, so this is a true "
                       "blind test of the saved model.")

    st.subheader("Log tracks")
    well_a = st.selectbox("Well", sorted(out['WELL_NAME'].unique()), key="apply_well")
    fig_a = plot_blind_well(out, well_a, pred_cols, target=tgt,
                            title=f"Saved-model {tgt} predictions: {well_a}")
    st.pyplot(fig_a)
    plt.close(fig_a)

    st.subheader("Export")
    st.caption(f"Predicted curves are named {tgt}_LR, {tgt}_RF, {tgt}_XGB, {tgt}_MLP.")
    st.download_button("Download predictions (CSV)",
                       for_export(out, tgt).to_csv(index=False).encode('utf-8'),
                       file_name=f"saved_model_{tgt}_predictions.csv", mime="text/csv",
                       key="apply_csv")
    if HAS_LASIO:
        depth_unit_a = st.selectbox("Depth unit for LAS header", ["FT", "M"],
                                    key="apply_depth_unit")
        try:
            st.download_button("Download ALL wells (ZIP of LAS files)",
                               build_las_zip(out, depth_unit_a, f"_{tgt}_predicted", tgt),
                               file_name=f"saved_model_{tgt}_predicted_las.zip",
                               mime="application/zip", key="apply_zip")
        except Exception as e:
            st.error("Could not build the ZIP of LAS files.")
            st.exception(e)
        try:
            st.download_button(f"Download LAS: {well_a}",
                               build_las_text(out[out['WELL_NAME'] == well_a], well_a,
                                              depth_unit_a, tgt).encode('utf-8'),
                               file_name=f"{safe_filename(well_a)}_{tgt}_predicted.las",
                               mime="text/plain", key=f"apply_las_{well_a}")
        except Exception as e:
            st.error(f"Could not build the LAS file for {well_a}.")
            st.exception(e)


# ---------------------------------------------------------------------------
# 5. SIDEBAR (user inputs)
# ---------------------------------------------------------------------------
st.title("🛢️ Well Log Prediction with Machine Learning")
st.caption("Predict compressional (DT) or shear (DTS) sonic from other logs, validated on "
           "wells the models never saw.")

with st.sidebar:
    mode = st.radio("Mode", ["Train & validate", "Apply saved model"],
                    help="Apply saved model: load a model file you saved earlier and "
                         "predict DT or DTS for new wells, without retraining.")
    st.divider()

if mode == "Apply saved model":
    if not HAS_SKLEARN:
        st.error("scikit-learn is not installed. Run `pip install scikit-learn`.")
    else:
        render_apply_mode()
    st.stop()

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Data source", ["Demo (synthetic wells)", "Upload LAS files"])

    uploaded = []
    if source == "Upload LAS files":
        if not HAS_LASIO:
            st.error("`lasio` is not installed. Run `pip install lasio`.")
        uploaded = st.file_uploader("LAS files (2+ wells)", type=["las", "LAS"],
                                    accept_multiple_files=True)

    st.header("2. Target & inputs")
    target = st.radio("Log to predict", list(TARGETS), format_func=TARGETS.get,
                      help="Switching target keeps separate input choices for each; "
                           "train and save a model for each target you need.")
    # For shear, the two input setups are an explicit, always-visible choice
    use_dt = False
    if target == 'DTS':
        dts_inputs = st.radio(
            "DTS input set",
            ["GR, Res (ILD), NPHI, RHOB",
             "GR, Res (ILD), NPHI, RHOB + DT"],
            key="dts_input_set",
            help="The second option adds the compressional sonic (DT) as an input. "
                 "It is usually more accurate, but depths without DT can't be "
                 "predicted. Train both and compare them in Results > Compare runs.")
        use_dt = dts_inputs.endswith("+ DT")

    base_curves = st.multiselect(
        "Input curves" if target == 'DT' else "Base input curves",
        options=['GR', 'RHOB', 'NPHI', 'ILD', 'CALI'],
        default=['GR', 'RHOB', 'NPHI', 'ILD'],
        key=f"features_{target}",
        help="ILD (deep resistivity) is converted to log10 automatically.")
    feature_choice = base_curves + (['DT'] if use_dt else [])
    if target == 'DTS':
        st.caption(f"Model inputs: **{', '.join(feature_choice) or 'none'}**")

    with st.expander("Cleaning limits (all logs)"):
        st.caption("Readings outside these ranges are removed from every well. "
                   "Applies to all logs, including ones not used as inputs, and to "
                   "DT / DTS.")
        limits = {}
        limits['GR'] = st.slider("GR range (API)", 0.0, 500.0, (0.0, 300.0), 5.0)
        limits['RHOB'] = st.slider("RHOB range (g/cc)", 1.0, 3.5, (1.8, 3.0), 0.05)
        limits['NPHI'] = st.slider("NPHI range (v/v)", -0.15, 1.0, (-0.05, 0.60), 0.01)
        c_lo, c_hi = st.columns(2)
        ild_min = c_lo.number_input("ILD min (ohm.m)", 0.01, 1e5, 0.1, format="%.2f")
        ild_max = c_hi.number_input("ILD max (ohm.m)", 0.01, 1e5, 2000.0, format="%.1f")
        limits['ILD'] = (ild_min, ild_max)
        limits['CALI'] = st.slider("CALI range (in)", 2.0, 30.0, (5.0, 20.0), 0.5)
        limits['DT'] = st.slider("DT range (us/ft)", 20.0, 700.0, (40.0, 200.0), 1.0,
                                 help="If your sonic is in us/m, widen this to about 130-660.")
        limits['DTS'] = st.slider("DTS range (us/ft)", 40.0, 1200.0, (60.0, 600.0), 10.0,
                                  help="Shear slowness. If it is in us/m, widen this to "
                                       "about 200-2000.")

        st.markdown("**Spike filter (optional)**")
        spike_pct = st.slider(
            "Trim each tail, per well (%)", 0.0, 5.0, 0.0, 0.5,
            help="0 = off. E.g. 1% removes readings below the 1st and above the 99th "
                 "percentile of each curve, separately in each well. Catches spikes "
                 "that are still inside the physical range.")

    st.header("3. Validation")
    test_size = st.slider("Fraction of wells held out (blind)", 0.1, 0.5, 0.25, 0.05)

    st.header("4. Models")
    model_options = ['Linear Regression', 'Random Forest']
    if HAS_XGB:
        model_options.append('XGBoost')
    if HAS_TF:
        model_options.append('Deep Learning (MLP)')
    selected_models = st.multiselect("Models to train", model_options, default=model_options)

    missing = [n for n, ok in [('xgboost', HAS_XGB), ('tensorflow', HAS_TF)] if not ok]
    if missing:
        st.info(f"Not installed (hidden): {', '.join(missing)}")

    params = {}
    with st.expander("Hyperparameters"):
        params['rf_trees'] = st.slider("RF: trees", 50, 500, 100, 50)
        params['rf_depth'] = st.slider("RF: max depth", 3, 30, 10)
        params['xgb_trees'] = st.slider("XGB: trees", 50, 500, 100, 50)
        params['xgb_lr'] = st.slider("XGB: learning rate", 0.01, 0.5, 0.08, 0.01)
        params['xgb_depth'] = st.slider("XGB: max depth", 2, 12, 5)
        params['mlp_epochs'] = st.slider("MLP: max epochs", 10, 200, 30, 10)

    run = st.button("Train models", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# 6. MAIN PANEL
# ---------------------------------------------------------------------------
# Load data -----------------------------------------------------------------
df_raw, load_errors = None, []
if source == "Demo (synthetic wells)":
    df_raw = simulate_demo_data()
elif uploaded and HAS_LASIO:
    payload = tuple((f.name, f.getvalue()) for f in uploaded)
    df_raw, load_errors = load_uploaded_las(payload)

for err in load_errors:
    st.warning(f"Could not read {err}")

if df_raw is None:
    st.info("Upload two or more LAS files in the sidebar, or switch to the demo data.")
    st.stop()

# Validate selections -------------------------------------------------------
T_NAME = TARGETS[target]
if target not in df_raw.columns or df_raw[target].isna().all():
    # Still allow cleaning + export (Data tab); there is just nothing to train on
    st.warning(f"No {T_NAME} curve found in these wells, so {target} models can't be "
               "trained. You can still clean and export the logs in the Data tab, or use "
               "**Apply saved model** mode. If the curve has another name, add its "
               "mnemonic to `LAS_ALIASES`.")
    if target not in df_raw.columns:
        df_raw = df_raw.assign(**{target: np.nan})

# Only offer curves that actually exist in the loaded data
available = [c for c in feature_choice if c in df_raw.columns]
dropped = [c for c in feature_choice if c not in df_raw.columns]
if dropped:
    st.warning(f"Selected curves not found in data and ignored: {', '.join(dropped)}")
if not available:
    st.error("Select at least one input curve that exists in the data.")
    st.stop()

# Model features: ILD is swapped for its log10 version
model_features = ['ILD_log10' if c == 'ILD' else c for c in available]

if limits['ILD'][0] >= limits['ILD'][1]:
    st.error("ILD min must be smaller than ILD max (Cleaning limits in the sidebar).")
    st.stop()

df_clean, clean_report = clean_data(df_raw, available, limits, spike_pct,
                                    return_report=True, target=target)

# Results are kept per target, so DT and DTS runs don't overwrite each other
RES_KEY, FINAL_KEY = f'results_{target}', f'final_{target}'

tab_data, tab_results, tab_plot, tab_final = st.tabs(
    ["📋 Data", "📊 Results", "📈 Log plot", "🌍 All wells (final model)"])

# Data tab ------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Wells with {target} (training)", df_clean['WELL_NAME'].nunique())
    c2.metric("Rows (raw)", f"{len(df_raw):,}")
    c3.metric("Rows (after cleaning)", f"{len(df_clean):,}")

    st.subheader("Outlier cleaning per log")
    st.dataframe(clean_report, use_container_width=True)
    st.caption("Counts are individual readings set to null. A row is only dropped from "
               f"training if it loses a selected input curve or its {target}.")

    st.subheader(f"Training rows per well (inputs + {target} all valid)")
    st.bar_chart(df_clean['WELL_NAME'].value_counts())

    st.subheader("Cleaned data preview")
    st.dataframe(df_clean.head(200), use_container_width=True)

    # Export cleaned logs (no training needed) -----------------------------
    st.subheader("Export cleaned logs")
    st.caption("All logs after outlier cleaning, using the current limits in the sidebar. "
               "Every depth is kept; readings removed as outliers are written as null "
               "(-999.25 in LAS). No training is needed for this.")
    include_other = st.checkbox(
        "Include other curves from the LAS files (not outlier-cleaned)", value=False,
        help="Off: only GR, RHOB, NPHI, ILD, CALI, DT and DTS. On: also curves like PEF "
             "or DRHO, exported exactly as read.")
    df_export = cleaned_logs_for_export(df_raw, limits, spike_pct, include_other)
    st.write(f"{df_export['WELL_NAME'].nunique()} wells, {len(df_export):,} rows, "
             f"curves: {', '.join(df_export.columns[2:])}")

    st.download_button("Download cleaned logs (CSV)",
                       df_export.to_csv(index=False).encode('utf-8'),
                       file_name="cleaned_logs.csv", mime="text/csv", key="clean_csv")
    if not HAS_LASIO:
        st.warning("Install `lasio` to export LAS files.")
    else:
        e1, e2 = st.columns(2)
        depth_unit_c = e1.selectbox("Depth unit for LAS header", ["FT", "M"],
                                    key="clean_depth_unit",
                                    help="Written into the DEPT curve header only; "
                                         "values are not converted.")
        well_c = e2.selectbox("Single well", sorted(df_export['WELL_NAME'].unique()),
                              key="clean_well")
        try:
            st.download_button("Download ALL wells (ZIP of cleaned LAS files)",
                               build_las_zip(df_export, depth_unit_c, suffix="_cleaned"),
                               file_name="cleaned_las.zip", mime="application/zip",
                               key="clean_zip")
        except Exception as e:
            st.error("Could not build the ZIP of cleaned LAS files.")
            st.exception(e)
        try:
            st.download_button(f"Download cleaned LAS: {well_c}",
                               build_las_text(df_export[df_export['WELL_NAME'] == well_c],
                                              well_c, depth_unit_c).encode('utf-8'),
                               file_name=f"{safe_filename(well_c)}_cleaned.las",
                               mime="text/plain", key=f"clean_las_{well_c}")
        except Exception as e:
            st.error(f"Could not build the cleaned LAS file for {well_c}.")
            st.exception(e)

# Train on button press -----------------------------------------------------
if run:
    n_wells = df_clean['WELL_NAME'].nunique()
    if n_wells < 2:
        st.error("Need at least 2 wells so one can be held out as the blind test.")
    elif not selected_models:
        st.error("Select at least one model.")
    else:
        with st.spinner(f"Training {target} models..."):
            # Keep results in session_state so they survive widget interaction
            # (Streamlit reruns the whole script on every change).
            res = train_and_predict(df_clean, model_features, selected_models,
                                    test_size, params, target=target)
            st.session_state[RES_KEY] = res
            # Log every run so different input sets can be compared
            run_log = st.session_state.setdefault('run_log', [])
            run_id = 1 + max((r['Run'] for r in run_log), default=0)
            for model_label, row in res['metrics'].iterrows():
                run_log.append({
                    'Run': run_id,
                    'Target': target,
                    'Inputs': ', '.join(available),
                    'Model': model_label,
                    'RMSE (us/ft)': row['RMSE (us/ft)'],
                    'MAE (us/ft)': row['MAE (us/ft)'],
                    'R²': row['R²'],
                    'Blind wells': ', '.join(res['test_wells']),
                })

# Results tab ---------------------------------------------------------------
results = st.session_state.get(RES_KEY)
if results is not None and results['features'] != model_features:
    results = None  # trained with other inputs; don't mix them up

with tab_results:
    if results is None:
        st.info(f"Configure options in the sidebar and click **Train models** to train "
                f"{target} models with the current inputs.")
    else:
        st.write(f"**Predicting:** {T_NAME}")
        st.write(f"**Training wells:** {', '.join(results['train_wells'])}")
        st.write(f"**Blind test wells:** {', '.join(results['test_wells'])}")
        st.write(f"**Inputs:** {', '.join(available)}")

        st.subheader("Blind-well performance")
        st.dataframe(
            results['metrics'].style.format({'RMSE (us/ft)': '{:.2f}',
                                             'MAE (us/ft)': '{:.2f}',
                                             'R²': '{:.3f}'})
                                    .highlight_min(subset=['RMSE (us/ft)'], color='#c8e6c9')
                                    .highlight_max(subset=['R²'], color='#c8e6c9'),
            use_container_width=True)

        if len(results['test_wells']) == 1:
            st.caption("Only one blind well - scores can swing a lot depending on which "
                       "well is held out. Add more wells for a more reliable estimate.")
        if source == "Demo (synthetic wells)":
            st.caption("Demo data has a built-in DT / DTS relationship, so high scores "
                       "here say nothing about real-world accuracy.")

        if results['history'] is not None:
            st.subheader("MLP training curve")
            st.line_chart(pd.DataFrame({'train loss': results['history']['loss'],
                                        'validation loss': results['history']['val_loss']}))

        st.subheader("Export")
        csv = for_export(results['blind_df'], target).to_csv(index=False).encode('utf-8')
        st.download_button("Download blind-well predictions (CSV)", csv,
                           file_name=f"blind_well_{target}_predictions.csv", mime="text/csv")

        # LAS export: one file per blind well, predictions added as new curves
        if not HAS_LASIO:
            st.warning("LAS export is off because `lasio` is not installed in the Python "
                       "running this app. Run: `pip install lasio` in that environment.")
        else:
            depth_unit = st.selectbox("Depth unit for LAS header", ["FT", "M"],
                                      help="Written into the DEPT curve header only; "
                                           "values are not converted.")
            st.caption(f"Each LAS contains the input logs, measured {target}, and predicted "
                       f"curves ({target}_LR, {target}_RF, {target}_XGB, {target}_MLP) for one blind well.")
            for well in results['test_wells']:
                well_df = results['blind_df'][results['blind_df']['WELL_NAME'] == well]
                try:
                    las_text = build_las_text(well_df, well, depth_unit, target)
                    st.download_button(f"Download LAS: {well}",
                                       las_text.encode('utf-8'),
                                       file_name=f"{safe_filename(well)}_{target}_predicted.las",
                                       mime="text/plain",
                                       key=f"las_{target}_{well}")  # unique key per button
                except Exception as e:
                    st.warning(f"Could not build LAS for {well}: {e}")

    # Comparison of every training run in this session ----------------------
    run_log = st.session_state.get('run_log', [])
    if run_log:
        st.subheader("Compare runs")
        st.caption("Every **Train models** run in this session, e.g. DTS with and without "
                   "DT as an input. Compare runs that share the same blind wells; a "
                   "different blind well can change the scores on its own.")
        log_df = pd.DataFrame(run_log)
        only_this = st.checkbox(f"Only {target} runs", value=True, key="log_filter")
        if only_this:
            log_df = log_df[log_df['Target'] == target]
        st.dataframe(log_df.style.format({'RMSE (us/ft)': '{:.2f}', 'MAE (us/ft)': '{:.2f}',
                                          'R²': '{:.3f}'}),
                     use_container_width=True, hide_index=True)
        if st.button("Clear run history", key="clear_log"):
            st.session_state['run_log'] = []
            st.rerun()

# Plot tab ------------------------------------------------------------------
with tab_plot:
    if results is None:
        st.info("Train the models first to see the log tracks.")
    else:
        well = st.selectbox("Blind well", results['test_wells'])
        fig = plot_blind_well(results['blind_df'], well, results['pred_cols'], target=target,
                              title=f"Blind {target} prediction: {well}")
        st.pyplot(fig)
        plt.close(fig)  # free memory; Streamlit reruns create many figures

# All-wells / final model tab -------------------------------------------------
with tab_final:
    st.markdown(
        f"Train **one final model on every well that has a measured {target}**, then "
        f"predict {target} for **all** wells, including wells where it is missing or only "
        "partly logged. There is no hold-out here, so judge accuracy from the blind-well "
        "validation in the **Results** tab.")
    if 'DT' in available:
        st.caption("DT is an input, so only depths with a valid DT can be predicted. To "
                   "cover wells without DT, predict DT first, then use **Apply saved model** "
                   "with the exported LAS files to fill the gaps with predicted DT.")

    # Rows we can predict: valid input curves, target NOT required
    df_predict = clean_data(df_raw, available, limits, spike_pct, require_target=False,
                            target=target)
    n_train_wells = df_clean['WELL_NAME'].nunique()

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Wells with measured {target} (training)", n_train_wells)
    m2.metric("Training rows", f"{len(df_clean):,}")
    m3.metric(f"Rows predicted without a measured {target}",
              f"{len(df_predict) - len(df_clean):,}")

    if results is None:
        st.info("Tip: run **Train models** in the sidebar first, so you know how accurate "
                "this setup is on unseen wells before relying on the final model.")
    else:
        st.caption("Blind-well validation from the Results tab (the honest accuracy estimate):")
        st.dataframe(results['metrics'].round(3), use_container_width=True)

    if n_train_wells == 1:
        st.warning(f"Only one well has a measured {target}, so the final model will only "
                   "know that one well's geology. Predictions elsewhere may be unreliable.")

    run_final = st.button(f"Train final {target} model & predict all wells", type="primary",
                          key="run_final")
    if run_final:
        if len(df_clean) == 0:
            st.error(f"No rows with a measured {target} and valid inputs to train on.")
        elif not selected_models:
            st.error("Select at least one model in the sidebar.")
        else:
            with st.spinner(f"Training final {target} model on all wells..."):
                final_new = train_final_and_predict_all(
                    df_clean, df_predict, model_features, selected_models, params,
                    target=target)

            # Package it for saving now, while the sidebar settings that
            # produced it are still the current ones.
            blind = None
            if results is not None:  # already checked: same target and inputs
                blind = results['metrics'].round(4).to_dict('index')
            meta = {
                'app': 'well_log_ml_app', 'bundle_version': BUNDLE_VERSION,
                'created': datetime.now().isoformat(timespec='seconds'),
                'target': target,
                'input_curves': available,           # raw curve names needed
                'model_features': model_features,    # columns fed to the scaler
                'limits': {k: [float(v[0]), float(v[1])] for k, v in limits.items()},
                'spike_pct': float(spike_pct),
                'models': {c: lab for lab, c in final_new['pred_cols'].items()},
                'hyperparameters': params,
                'train_wells': sorted(df_clean['WELL_NAME'].unique()),
                'n_train_rows': int(len(df_clean)),
                # 1st-99th percentile of each input: flags extrapolation later
                'feature_ranges': {f: [float(df_clean[f].quantile(0.01)),
                                       float(df_clean[f].quantile(0.99))]
                                   for f in model_features},
                'blind_validation': blind,
            }
            try:
                meta['versions'] = _versions()
                final_new['bundle'] = make_model_bundle(final_new, meta)
            except Exception as e:
                final_new['bundle'] = None
                final_new['bundle_error'] = e
            st.session_state[FINAL_KEY] = final_new

    final = st.session_state.get(FINAL_KEY)
    if final is None:
        st.info(f"Click the button above to train the final {target} model.")
    else:
        out = final['predictions']

        if final['features'] != model_features:
            st.warning("The input curves have changed since this final model was trained. "
                       "Click the button again to retrain. (Shown below: the model trained "
                       f"with {', '.join(final['features'])}.)")

        st.write(f"**Predicting:** {T_NAME}")
        st.write(f"**Inputs:** {', '.join(final['features'])}")
        st.write(f"**Models:** {', '.join(final['pred_cols'])}")

        # Wells that had no usable rows at all (e.g. all input curves missing)
        skipped = sorted(set(df_raw['WELL_NAME']) - set(out['WELL_NAME']))
        if skipped:
            st.warning("No predictions for these wells (inputs missing or out of range): "
                       + ", ".join(skipped))

        st.subheader("Predictions per well")
        st.dataframe(final['summary'], use_container_width=True, hide_index=True)

        st.subheader("In-sample fit (training rows)")
        st.dataframe(final['in_sample'].round(3), use_container_width=True)
        st.caption("These scores are measured on data the model was trained on, so they "
                   "are optimistic and NOT a performance estimate.")

        if final['history'] is not None:
            st.subheader("MLP training curve (final model)")
            st.line_chart(pd.DataFrame({'train loss': final['history']['loss'],
                                        'validation loss': final['history']['val_loss']}))

        # Log tracks for any well -------------------------------------------
        st.subheader("Log tracks")
        well_f = st.selectbox("Well", sorted(out['WELL_NAME'].unique()), key="final_well")
        n_meas = int(out.loc[out['WELL_NAME'] == well_f, 'USED_IN_TRAINING'].sum())
        if n_meas > 0:
            st.caption(f"{n_meas:,} rows of this well were in the training data, so the "
                       "predictions there are in-sample.")
        else:
            st.caption(f"This well has no measured {target}; the curves shown are "
                       "predictions only.")
        fig_f = plot_blind_well(out, well_f, final['pred_cols'], target=target,
                                title=f"Final-model {target} predictions: {well_f}")
        st.pyplot(fig_f)
        plt.close(fig_f)

        # Export -------------------------------------------------------------
        st.subheader("Export")
        st.download_button("Download all-well predictions (CSV)",
                           for_export(out, target).to_csv(index=False).encode('utf-8'),
                           file_name=f"all_wells_{target}_predictions.csv", mime="text/csv",
                           key="final_csv")

        if not HAS_LASIO:
            st.warning("LAS export is off because `lasio` is not installed in the Python "
                       "running this app. Run: `pip install lasio` in that environment.")
        else:
            depth_unit_f = st.selectbox("Depth unit for LAS header", ["FT", "M"],
                                        key="final_depth_unit",
                                        help="Written into the DEPT curve header only; "
                                             "values are not converted.")
            st.caption(f"LAS files contain the input logs, measured {target} (where it "
                       f"exists) and predicted curves ({target}_LR, {target}_RF, {target}_XGB, "
                       f"{target}_MLP). Missing values are written as -999.25.")
            # Two separate try blocks: if one export fails, the other button
            # still appears, and the full error is shown instead of hidden.
            try:
                st.download_button("Download ALL wells (ZIP of LAS files)",
                                   build_las_zip(out, depth_unit_f, f"_{target}_predicted",
                                                 target),
                                   file_name=f"all_wells_{target}_predicted_las.zip",
                                   mime="application/zip", key="final_zip")
            except Exception as e:
                st.error("Could not build the ZIP of LAS files.")
                st.exception(e)

            try:
                las_one = build_las_text(out[out['WELL_NAME'] == well_f], well_f,
                                         depth_unit_f, target)
                st.download_button(f"Download LAS: {well_f}",
                                   las_one.encode('utf-8'),
                                   file_name=f"{safe_filename(well_f)}_{target}_predicted.las",
                                   mime="text/plain", key=f"final_las_{target}_{well_f}")
            except Exception as e:
                st.error(f"Could not build the LAS file for {well_f}.")
                st.exception(e)

        # Save model -----------------------------------------------------------
        st.subheader("Save model")
        st.caption("Saves every model trained above, plus the scaler, input curves and "
                   "cleaning limits, in one .zip file. Load it later in **Apply saved "
                   f"model** mode (top of the sidebar) to predict {target} for new wells; "
                   "you choose which model to apply there.")
        if final.get('bundle') is None:
            st.error("Could not package the model for saving.")
            if final.get('bundle_error') is not None:
                st.exception(final['bundle_error'])
        else:
            default_name = (f"{target.lower()}_model"
                            f"{'_with_dt' if 'DT' in final['features'] else ''}"
                            f"_{datetime.now():%Y%m%d}")
            model_name = st.text_input("File name", value=default_name,
                                       key=f"model_name_{default_name}")
            st.download_button("💾 Save model file",
                               final['bundle'],
                               file_name=f"{safe_filename(model_name) or default_name}.zip",
                               mime="application/zip", key="save_model")
            st.caption(f"Size: {len(final['bundle']) / 1e6:.1f} MB. Large Random Forests "
                       "make large files; fewer trees or a smaller max depth shrinks them.")
