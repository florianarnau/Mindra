# app.py — Mindra : Pipeline ML multi-classe
import io
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =============================================================================
# Streamlit config
# =============================================================================
st.set_page_config(
    page_title="Mindra — Pipeline ML multi-classe",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.image("logo.png", width=160)
st.title("Pipeline ML multi-classe")
st.caption("Exploration → Préparation → Modélisation → Évaluation")

# =============================================================================
# Constantes & utilitaires
# =============================================================================
MISSING_TOKENS = [
    "", " ", "  ", "\t", "NA", "N/A", "na", "n/a", "Null", "NULL", "null",
    "None", "NONE", "none", "?", "-", "--"
]

# Stopwords FR (liste) pour scikit-learn (qui n’accepte que 'english' en str)
FRENCH_STOPWORDS = [
    "a", "à", "â", "afin", "ai", "ainsi", "après", "assez", "au", "aucun", "aussi", "autre", "avant", "avec", "avoir",
    "bon", "c", "car", "ce", "cela", "ces", "cet", "cette", "ceci", "ceux", "chaque", "ci", "comme", "comment", "d",
    "dans", "de", "des", "du", "dont", "déjà", "elle", "elles", "en", "encore", "est", "et", "étaient", "était",
    "être", "eu", "fait", "faut", "hors", "il", "ils", "je", "jusqu", "l", "la", "le", "les", "leur", "là",
    "ma", "mais", "mal", "me", "même", "mes", "mien", "moins", "mon", "ne", "ni", "nos", "notre", "nous", "on", "ou",
    "où", "par", "parce", "pas", "peu", "peut", "plus", "pour", "pourquoi", "qu", "quand", "que", "quel", "quelle",
    "quelles", "quels", "qui", "sa", "sans", "se", "ses", "si", "sien", "son", "sont", "sur", "t", "ta", "te", "tes",
    "toi", "ton", "tous", "tout", "très", "tu", "un", "une", "vos", "votre", "vous", "y"
]

@st.cache_data
def load_csv(upload) -> pd.DataFrame:
    """
    Lecture CSV + standardisation agressive des NaN + trim objets + drop index fantôme.
    """
    df = pd.read_csv(upload, na_values=MISSING_TOKENS, keep_default_na=True)
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        df[obj_cols] = (
            df[obj_cols]
            .apply(lambda col: col.astype(str).str.strip())
            .replace(MISSING_TOKENS, np.nan)
            .replace(r"^\s*$", np.nan, regex=True)
        )
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.drop(columns=[df.columns[0]])
    return df

def truncate_rare_list(rare_series: pd.Series, max_items: int = 10) -> str:
    items = [f"{cls} ({int(n)})" for cls, n in rare_series.items()]
    if len(items) > max_items:
        return ", ".join(items[:max_items]) + f" … (+{len(items)-max_items})"
    return ", ".join(items)

def infer_types(
    df: pd.DataFrame, target_name: str, text_threshold_unique: float = 0.5, text_threshold_len: int = 25
) -> Tuple[List[str], List[str], List[str]]:
    """
    Renvoie (num_cols, cat_cols, text_cols) hors cible, avec heuristique texte :
    - texte si forte cardinalité ou chaînes longues
    """
    candidates = [c for c in df.columns if c != target_name]
    num_cols, cat_cols, text_cols = [], [], []
    for c in candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            uniq = s.nunique(dropna=True)
            unique_ratio = uniq / max(1, len(s))
            avg_len = s.dropna().astype(str).str.len().mean() if len(s.dropna()) else 0
            if (unique_ratio >= text_threshold_unique) or (avg_len >= text_threshold_len):
                text_cols.append(c)
            else:
                cat_cols.append(c)
    return num_cols, cat_cols, text_cols

def iqr_winsorize(series: pd.Series, iqr_k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - iqr_k * iqr, upper=q3 + iqr_k * iqr)

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorisation IQR pour colonnes numériques."""
    def __init__(self, cols=None, iqr_k: float = 1.5):
        self.cols = cols or []
        self.iqr_k = float(iqr_k)
    def fit(self, X, y=None):
        self.cols_ = list(self.cols)
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X, columns=self.cols_)
        for c in self.cols_:
            Xdf[c] = iqr_winsorize(Xdf[c], self.iqr_k)
        return Xdf.values

class Densifier(BaseEstimator, TransformerMixin):
    """Convertit sparse→dense pour estimateurs qui l'exigent / accélèrent en dense."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X

class TextJoiner(BaseEstimator, TransformerMixin):
    """Joint N colonnes texte en une seule chaîne avant TF-IDF."""
    def __init__(self, sep: str = " "):
        self.sep = sep
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X)
        # remplace "nan" textuel par vide, puis join
        joined = Xdf.astype(str).replace("nan", "", regex=False).apply(
            lambda r: self.sep.join([v for v in r if v]), axis=1
        )
        return joined.values

def build_preprocessor(
    num_cols, cat_cols, text_cols,
    scaling, impute_num, impute_cat, outlier_mode, outlier_iqr_k,
    ohe_min_freq, tfidf_max_features, tfidf_min_df, tfidf_stop_lang
) -> ColumnTransformer:
    scalers = {
        "Aucun": "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }
    scaler = scalers.get(scaling, "passthrough")

    # Numérique
    num_steps = []
    if outlier_mode == "Winsorisation (IQR)" and len(num_cols) > 0:
        num_steps.append(("winsor", OutlierClipper(cols=num_cols, iqr_k=outlier_iqr_k)))
    num_steps.append(("imputer", SimpleImputer(strategy=impute_num)))
    num_steps.append(("scaler", scaler))

    # Catégorielles (regroupement rares via min_frequency)
    cat_steps = [
        ("imputer", SimpleImputer(strategy=impute_cat, fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore",
                              min_frequency=ohe_min_freq if ohe_min_freq and ohe_min_freq > 0 else None)),
    ]

    # Texte : impute "" + join + TF-IDF (stopwords FR via liste)
    if tfidf_stop_lang == "english":
        stop = "english"
    elif tfidf_stop_lang == "french":
        stop = FRENCH_STOPWORDS
    else:
        stop = None
    text_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("join", TextJoiner(sep=" ")),
        ("tfidf", TfidfVectorizer(max_features=int(tfidf_max_features),
                                  min_df=int(tfidf_min_df),
                                  stop_words=stop)),
    ])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
    if len(text_cols) > 0:
        transformers.append(("txt", text_pipeline, text_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.1)

def make_scatter_matrix(df: pd.DataFrame, cols: List[str], color_by: Optional[str] = None):
    base = alt.Chart(df)
    rows = []
    for y in cols:
        row = []
        for x in cols:
            ch = base.mark_circle(opacity=0.6, size=30).encode(
                x=alt.X(x, type="quantitative", title=x),
                y=alt.Y(y, type="quantitative", title=y),
                tooltip=list(set([x, y] + ([color_by] if color_by else []))),
            )
            if color_by:
                ch = ch.encode(color=alt.Color(color_by, type="nominal"))
            row.append(ch)
        rows.append(alt.hconcat(*row).resolve_scale(color="independent"))
    return alt.vconcat(*rows).interactive()

def confusion_chart(cm: np.ndarray, labels: List[str]) -> alt.Chart:
    df_cm = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(id_vars="index")
    df_cm.columns = ["Vérité", "Prédiction", "Nb"]
    heat = alt.Chart(df_cm).mark_rect().encode(
        x=alt.X("Prédiction:N", sort=labels),
        y=alt.Y("Vérité:N", sort=labels),
        color=alt.Color("Nb:Q"),
        tooltip=["Vérité", "Prédiction", "Nb"],
    )
    txt = alt.Chart(df_cm).mark_text().encode(x="Prédiction:N", y="Vérité:N", text="Nb:Q")
    return heat + txt

def stratify_feasible(y: pd.Series, test_size: float) -> bool:
    counts = y.value_counts()
    if counts.min() < 2:
        return False
    for n in counts.values:
        if int(np.floor(n * test_size)) < 1:
            return False
        if int(np.floor(n * (1 - test_size))) < 1:
            return False
    return True

# =============================================================================
# Chargement & préparation initiale
# =============================================================================
st.sidebar.header("1) Données")
uploaded = st.sidebar.file_uploader("Importer un CSV (séparateur: comma)", type=["csv"])
if uploaded is None:
    st.info("Aucun fichier importé.")
    st.stop()

df = load_csv(uploaded)

# Nettoyage avancé
st.sidebar.header("Nettoyage avancé (optionnel)")
col_drop = st.sidebar.slider("Supprimer colonnes avec > x% de NaN", 0, 90, 0, 5)
row_drop = st.sidebar.slider("Supprimer lignes avec > x% de NaN", 0, 90, 0, 5)
if col_drop > 0:
    bad_cols = df.columns[(df.isna().mean() * 100) > col_drop].tolist()
    if len(bad_cols) > 0:
        st.sidebar.warning(f"Colonnes supprimées (NaN > {col_drop}%): {', '.join(bad_cols)}")
        df = df.drop(columns=bad_cols)
if row_drop > 0:
    before = len(df)
    df = df[(df.isna().mean(axis=1) * 100) <= row_drop].reset_index(drop=True)
    st.sidebar.info(f"Lignes supprimées: {before - len(df)}")

# Cible & exclusions
all_cols = df.columns.tolist()
default_target = "target" if "target" in all_cols else ("Churn" if "Churn" in all_cols else all_cols[-1])
target_col = st.sidebar.selectbox("Colonne cible (classe)", options=all_cols, index=all_cols.index(default_target))

exclude_cols = st.sidebar.multiselect(
    "Colonnes à exclure des features",
    options=[c for c in all_cols if c != target_col],
    default=[c for c in ["customerID", "id", "ID"] if c in all_cols],
)

# Discrétisation éventuelle de la cible
st.sidebar.markdown("---")
discretize_numeric_target = st.sidebar.checkbox(
    "Discrétiser la cible si numérique", value=False,
    help="Transforme une cible continue en classes (bins) pour rester en classification."
)
nb_bins = st.sidebar.slider("Nombre de classes (bins)", 2, 7, 3, 1)
binning_method = st.sidebar.selectbox("Méthode de binning", ["quantiles", "largeur égale"], index=0)

# Build X, y
y_raw = df[target_col]
X = df.drop(columns=[target_col] + exclude_cols)

# Coercition numériques déguisés (70%+ parsables) + virgule→point
auto_coerce = st.sidebar.checkbox("Convertir automatiquement les numériques déguisés", value=True)
if auto_coerce:
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        s = X[c].astype(str).str.replace(",", ".", regex=False)
        # nombre (optionnel sci) strict
        mask = s.str.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", na=False, case=False)
        if mask.mean() >= 0.7:
            X[c] = pd.to_numeric(s, errors="coerce")

# Discrétiser si souhaité
if pd.api.types.is_numeric_dtype(y_raw) and discretize_numeric_target:
    try:
        if binning_method == "quantiles":
            y = pd.qcut(y_raw, q=nb_bins, duplicates="drop")
        else:
            y = pd.cut(y_raw, bins=nb_bins)
        st.sidebar.success(f"Cible discrétisée en {y.nunique()} classes.")
    except Exception as e:
        st.sidebar.error(f"Échec de la discrétisation: {e}")
        y = y_raw
else:
    y = y_raw

# Retirer les lignes où la cible est NaN
if y.isna().any():
    nmiss = int(y.isna().sum())
    st.warning(f"La cible contient {nmiss} valeur(s) manquante(s) : ces lignes seront exclues.")
    mask_valid = y.notna()
    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)

# Inférence des types (incluant texte)
num_cols, cat_cols, text_cols = infer_types(pd.concat([X, y], axis=1), y.name)
class_counts = y.value_counts()

# Alerte cible continue non discrétisée
if pd.api.types.is_numeric_dtype(y) and not discretize_numeric_target and y.nunique() > 20:
    st.sidebar.warning(
        f"La cible '{target_col}' a {y.nunique()} valeurs distinctes (continue). "
        "Active la discrétisation pour rester en classification."
    )

# =============================================================================
# Onglets
# =============================================================================
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(
    ["Traitement des données", "Visualisations", "Modélisation", "Évaluation"]
)

# === TAB 1 : Exploration & Préparation =======================================
with tabs_1:
    st.subheader("Exploration & Préparation")

    st.markdown("### Aperçu (brut après standardisation)")
    st.dataframe(df.head(50), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Échantillons", len(df))
    c2.metric("Variables totales", df.shape[1])
    c3.metric("Numériques", len(num_cols))
    c4.metric("Catégorielles", len(cat_cols))
    c5.metric("Texte", len(text_cols))

    st.markdown("### Valeurs manquantes (colonnes)")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].to_frame("Nb_NA")
    if not miss_df.empty:
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.success("Aucune valeur manquante détectée !")

    st.markdown("### Réglages de préparation")
    with st.expander("Imputation, normalisation, outliers, texte, sélection de variables"):
        cA, cB = st.columns(2)
        scaling = cA.selectbox("Normalisation", ["Aucun", "StandardScaler", "MinMaxScaler", "RobustScaler"], index=1)
        impute_num = cA.selectbox("Imputation numériques", ["mean", "median", "most_frequent"], index=1)
        impute_cat = cA.selectbox("Imputation catégorielles", ["most_frequent", "constant"], index=0)
        outlier_mode = cB.selectbox("Gestion des outliers (num.)", ["Aucun", "Winsorisation (IQR)"], index=1)
        outlier_iqr_k = cB.slider("Seuil IQR (k)", 0.5, 3.0, 1.5, 0.1)

        ohe_min_freq = cA.slider("Regrouper catégories rares (min_frequency OHE, proportion)",
                                 0.0, 0.05, 0.01, 0.005)
        tfidf_max_features = cA.number_input("TF-IDF max_features", 500, 20000, 5000, 500)
        tfidf_min_df = cA.number_input("TF-IDF min_df", 1, 50, 2, 1)
        tfidf_stop_lang = cB.selectbox("Stopwords", ["none", "english", "french"], index=2)

        sel_method = cB.selectbox("Sélection de variables", ["Aucune", "VarianceThreshold", "SelectKBest (ANOVA)"], index=0)
        k_best = cB.slider("k (si SelectKBest)", 1, 1000, 50, 1)

    st.session_state["prep_params"] = dict(
        scaling=scaling, impute_num=impute_num, impute_cat=impute_cat,
        outlier_mode=outlier_mode, outlier_iqr_k=float(outlier_iqr_k),
        ohe_min_freq=float(ohe_min_freq),
        tfidf_max_features=int(tfidf_max_features), tfidf_min_df=int(tfidf_min_df), tfidf_stop_lang=tfidf_stop_lang,
        sel_method=sel_method, k_best=int(k_best),
    )

    rare = class_counts[class_counts < 2]
    if not rare.empty:
        st.error("⚠️ Classe(s) ultra-rare(s) : " + truncate_rare_list(rare)
                 + ". Le split stratifié et la CV peuvent échouer.")

    with st.expander("Vérifier l'imputation/encodage sur un échantillon"):
        try:
            prep = st.session_state["prep_params"]
            preproc_preview = build_preprocessor(
                num_cols=[c for c in X.columns if c in num_cols],
                cat_cols=[c for c in X.columns if c in cat_cols],
                text_cols=[c for c in X.columns if c in text_cols],
                scaling=prep["scaling"], impute_num=prep["impute_num"], impute_cat=prep["impute_cat"],
                outlier_mode=prep["outlier_mode"], outlier_iqr_k=prep["outlier_iqr_k"],
                ohe_min_freq=prep["ohe_min_freq"], tfidf_max_features=prep["tfidf_max_features"],
                tfidf_min_df=prep["tfidf_min_df"], tfidf_stop_lang=prep["tfidf_stop_lang"],
            )
            sample = X.sample(min(300, len(X)), random_state=42)
            _ = preproc_preview.fit_transform(sample)
            st.success("Imputation/encodage OK sur l'échantillon (les NaN bruts sont traités dans le pipeline).")
            st.caption("Note : les colonnes encodées/TF-IDF changent d'espace de features et ne sont pas affichées.")
        except Exception as e:
            st.warning(f"Aperçu imputé indisponible : {e}")

# === TAB 2 : Visualisations ===================================================
with tabs_2:
    st.subheader("Visualisations interactives")
    df_plot = df if len(df) <= 10000 else df.sample(n=10000, random_state=42)

    num_only = [c for c in X.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    st.markdown("#### Distribution d’une variable")
    if len(num_only) > 0:
        col_to_plot = st.selectbox("Variable numérique à visualiser", options=num_only)
        if col_to_plot:
            if (not pd.api.types.is_numeric_dtype(y)) or (y.dtype.name == "category"):
                color_enc = alt.Color(f"{target_col}:N")
            else:
                color_enc = alt.value(None)
            hist = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X(f"{col_to_plot}:Q", bin=alt.Bin(maxbins=40)),
                y="count()", color=color_enc, tooltip=[col_to_plot, target_col],
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Aucune variable numérique détectée.")

    st.markdown("#### Matrice de corrélation (numériques)")
    if len(num_only) >= 2:
        corr = df[num_only].corr(numeric_only=True)
        corr_df = corr.reset_index().melt(id_vars="index")
        corr_df.columns = ["Var1", "Var2", "Corr"]
        heat = alt.Chart(corr_df).mark_rect().encode(
            x="Var2:N", y="Var1:N", color=alt.Color("Corr:Q"), tooltip=["Var1", "Var2", "Corr"]
        ).properties(height=400)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques pour une corrélation.")

    st.markdown("#### Scatter matrix (pairplot-like)")
    if len(num_only) >= 2:
        cols_select = st.multiselect("Colonnes numériques à inclure", num_only, default=num_only[:4])
        if len(cols_select) >= 2:
            sm = make_scatter_matrix(df_plot[[*cols_select, target_col]].dropna(), cols_select, color_by=target_col)
            st.altair_chart(sm, use_container_width=True)
        else:
            st.info("Sélectionne au moins 2 colonnes.")
    else:
        st.info("Pas assez de colonnes numériques pour un scatter matrix.")

    st.markdown("#### Répartition des classes")
    class_df = class_counts.reset_index()
    class_df.columns = ["Classe", "Nombre"]
    bar = alt.Chart(class_df).mark_bar().encode(
        x=alt.X("Classe:N", sort="-y"), y=alt.Y("Nombre:Q"), tooltip=["Classe", "Nombre"]
    ).properties(height=300)
    st.altair_chart(bar, use_container_width=True)

# === TAB 3 : Modélisation =====================================================
with tabs_3:
    st.subheader("Entraînement de modèles")

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Taille du test (%)", 10, 40, 20, 5) / 100.0
    random_state = c2.number_input("Random state", 0, 9999, 42, 1)
    do_cv = c3.checkbox("Validation croisée (jusqu’à 5-fold)", value=False)

    use_balanced = st.checkbox("Pondérer les classes (class_weight='balanced')", value=False)
    model_name = st.selectbox("Modèle", ["LogisticRegression", "RandomForest", "KNN", "SVM (RBF)"], index=1)

    with st.expander("Hyperparamètres"):
        if model_name == "LogisticRegression":
            C = st.slider("C (inverse régularisation)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("max_iter", 100, 2000, 500, 50)
            clf = LogisticRegression(C=C, max_iter=max_iter, multi_class="ovr",
                                     class_weight=("balanced" if use_balanced else None))
        elif model_name == "RandomForest":
            n_estimators = st.slider("n_estimators", 50, 500, 200, 10)
            max_depth = st.slider("max_depth", 1, 50, 10, 1)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         random_state=random_state, n_jobs=-1,
                                         class_weight=("balanced" if use_balanced else None))
        elif model_name == "KNN":
            n_neighbors = st.slider("n_neighbors", 1, 25, 5, 1)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            C = st.slider("C", 0.01, 10.0, 1.0, 0.01)
            gamma = st.selectbox("gamma", ["scale", "auto"], index=0)
            clf = SVC(C=C, gamma=gamma, probability=True, random_state=random_state,
                      class_weight=("balanced" if use_balanced else None))

    # Préprocesseur
    prep = st.session_state["prep_params"]
    preprocessor = build_preprocessor(
        num_cols=[c for c in X.columns if c in num_cols],
        cat_cols=[c for c in X.columns if c in cat_cols],
        text_cols=[c for c in X.columns if c in text_cols],
        scaling=prep["scaling"], impute_num=prep["impute_num"], impute_cat=prep["impute_cat"],
        outlier_mode=prep["outlier_mode"], outlier_iqr_k=prep["outlier_iqr_k"],
        ohe_min_freq=prep["ohe_min_freq"], tfidf_max_features=prep["tfidf_max_features"],
        tfidf_min_df=prep["tfidf_min_df"], tfidf_stop_lang=prep["tfidf_stop_lang"],
    )

    # Sélection de features
    selector = None
    if prep["sel_method"] == "VarianceThreshold":
        selector = VarianceThreshold(threshold=0.0)
    elif prep["sel_method"] == "SelectKBest (ANOVA)":
        selector = SelectKBest(score_func=f_classif, k=prep["k_best"])

    def needs_dense(name: str) -> bool:
        # RF supporte sparse mais dense peut être plus rapide selon TF-IDF/OHE
        return name in {"SVM (RBF)", "KNN", "RandomForest"}

    steps = [("preprocess", preprocessor)]
    if selector is not None:
        steps.append(("feature_select", selector))
    if needs_dense(model_name):
        steps.append(("densify", Densifier()))
    steps.append(("model", clf))
    model = Pipeline(steps=steps)

    # Split robuste
    can_stratify = stratify_feasible(y, test_size)
    if not can_stratify:
        st.warning("Stratification impossible (classes trop petites). `stratify=y` désactivé.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if can_stratify else None
    )

    with st.expander("Répartition des classes après split"):
        def _counts(s: pd.Series) -> pd.Series:
            return pd.Series(s.value_counts().to_dict(), name="count")
        dist = pd.DataFrame({"train": _counts(y_train), "test": _counts(y_test)}).fillna(0).astype(int)
        st.dataframe(dist)

    if st.button("Entraîner le modèle"):
        model.fit(X_train, y_train)
        st.success("Modèle entraîné !")

        st.session_state["trained_model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["class_labels"] = list(pd.Index(y.unique()).astype(str))

        # Overfitting check
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

        st.markdown("### Vérification overfitting")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Accuracy (train)", f"{train_acc:.3f}")
        cB.metric("Accuracy (test)", f"{test_acc:.3f}")
        cC.metric("F1 (train)", f"{train_f1:.3f}")
        cD.metric("F1 (test)", f"{test_f1:.3f}")
        gap = train_acc - test_acc
        if gap > 0.05:
            st.warning(f"Possible sur-apprentissage : écart de {gap:.2f}.")
        else:
            st.success("Pas d’overfitting significatif.")

        # CV sûre
        if do_cv:
            min_per_class = int(class_counts.min())
            cv_folds = max(2, min(5, min_per_class))
            if cv_folds >= 2:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
                st.write(f"Validation croisée (accuracy, {cv_folds}-fold) : **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**")
            else:
                st.info("CV désactivée (classes trop petites).")

        # Export pipeline
        buf = io.BytesIO()
        pickle.dump(model, buf)
        st.download_button(
            "Télécharger le pipeline entraîné (.pkl)",
            buf.getvalue(),
            file_name=f"pipeline_{model_name}.pkl",
            mime="application/octet-stream",
        )

        # Importances / coefficients
        with st.expander("Importance des variables / coefficients"):
            try:
                feat_names = None
                try:
                    feat_names = list(model.named_steps["preprocess"].get_feature_names_out())
                except Exception:
                    feat_names = None

                if "feature_select" in model.named_steps and hasattr(model.named_steps["feature_select"], "get_support"):
                    mask = model.named_steps["feature_select"].get_support()
                    if feat_names is not None:
                        feat_names = [n for n, keep in zip(feat_names, mask) if keep]

                est = model.named_steps["model"]
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    if feat_names is None:
                        feat_names = [f"feat_{i}" for i in range(len(imp))]
                    imp_df = pd.DataFrame({"Feature": feat_names[: len(imp)], "Importance": imp}) \
                               .sort_values("Importance", ascending=False).head(25)
                    chart = alt.Chart(imp_df).mark_bar().encode(
                        x=alt.X("Importance:Q"), y=alt.Y("Feature:N", sort="-x")
                    ).properties(height=420)
                    st.altair_chart(chart, use_container_width=True)

                elif hasattr(est, "coef_"):
                    coefs = est.coef_
                    coef_mean = np.mean(np.abs(coefs), axis=0)
                    if feat_names is None:
                        coef_df = pd.DataFrame({"Indice": range(len(coef_mean)), "Poids": coef_mean})
                        chart = alt.Chart(coef_df.sort_values("Poids", ascending=False).head(25)).mark_bar().encode(
                            x=alt.X("Poids:Q"), y=alt.Y("Indice:N", sort="-x")
                        ).properties(height=420)
                    else:
                        coef_df = pd.DataFrame({"Feature": feat_names[: len(coef_mean)], "Poids": coef_mean})
                        chart = alt.Chart(coef_df.sort_values("Poids", ascending=False).head(25)).mark_bar().encode(
                            x=alt.X("Poids:Q"), y=alt.Y("Feature:N", sort="-x")
                        ).properties(height=420)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Le modèle ne fournit pas d’importances/coefs exploitables.")
            except Exception as e:
                st.warning(f"Impossible d’afficher les importances: {e}")

# === TAB 4 : Évaluation =======================================================
with tabs_4:
    st.subheader("Évaluation du modèle")
    if "trained_model" not in st.session_state:
        st.info("Entraîne d’abord un modèle dans l’onglet *Modélisation*.")
    else:
        model = st.session_state["trained_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)
        if hasattr(model.named_steps["model"], "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        c2.metric("Precision (weighted)", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
        c3.metric("Recall (weighted)", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
        c4.metric("F1 (weighted)", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")

        st.markdown("### Matrice de confusion")
        labels = list(pd.Index(pd.unique(y_test)).astype(str))
        cm = confusion_matrix(pd.Series(y_test, dtype=str), pd.Series(y_pred, dtype=str), labels=labels)
        st.altair_chart(confusion_chart(cm, labels), use_container_width=True)

        st.markdown("### Rapport de classification")
        st.code(classification_report(y_test, y_pred, zero_division=0), language="text")

        if y_proba is not None and len(np.unique(y_test)) > 2:
            try:
                auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                st.metric("ROC AUC (macro, OvR)", f"{auc_macro:.3f}")
            except Exception:
                st.info("ROC AUC indisponible (probabilités/labels non conformes).")

st.markdown("---")
st.caption("© 2025 Mindra — Application de Machine Learning interactive")
