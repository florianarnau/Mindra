import io
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
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


# -----------------------------------------------------------------------------
# Config Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mindra — Pipeline ML multi-classe",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.image("logo.png", width=180)
st.title("Pipeline ML multi-classe")
st.caption("Exploration → Préparation → Modélisation → Évaluation")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)


def infer_types(df: pd.DataFrame, target_name: str) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if c != target_name and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != target_name and not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def iqr_winsorize(series: pd.Series, iqr_k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    return series.clip(lower=low, upper=high)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise (IQR) les colonnes numériques reçues par le ColumnTransformer."""
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


def build_preprocessor(
    num_cols, cat_cols, scaling, impute_num, impute_cat, outlier_mode, outlier_iqr_k
) -> ColumnTransformer:
    scalers = {
        "Aucun": "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }
    scaler = scalers.get(scaling, "passthrough")

    num_steps = []
    if outlier_mode == "Winsorisation (IQR)" and len(num_cols) > 0:
        num_steps.append(("winsor", OutlierClipper(cols=num_cols, iqr_k=outlier_iqr_k)))
    num_steps.extend([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("scaler", scaler),
    ])

    cat_steps = [
        ("imputer", SimpleImputer(strategy=impute_cat)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ],
        remainder="drop",
    )


def make_scatter_matrix(df: pd.DataFrame, cols: List[str], color_by: Optional[str] = None):
    base = alt.Chart(df)
    charts = []
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
        charts.append(alt.hconcat(*row).resolve_scale(color="independent"))
    return alt.vconcat(*charts).interactive()


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


def truncate_rare_list(rare_series: pd.Series, max_items: int = 10) -> str:
    items = [f"{cls} ({n})" for cls, n in rare_series.items()]
    if len(items) > max_items:
        return ", ".join(items[:max_items]) + f" … (+{len(items) - max_items})"
    return ", ".join(items)


# -----------------------------------------------------------------------------
# Chargement des données
# -----------------------------------------------------------------------------
st.sidebar.header("1) Données")
uploaded = st.sidebar.file_uploader("Importer un CSV (séparateur: comma)", type=["csv"])

if uploaded is None:
    st.info("Aucun fichier importé !")
    st.stop()

df = load_csv(uploaded)

# supprimer éventuelle colonne d'index
if df.columns[0] in ("", "Unnamed: 0"):
    df = df.drop(columns=[df.columns[0]])

# cible + options de discrétisation
all_cols = list(df.columns)
default_target = "target" if "target" in all_cols else all_cols[-1]
target_col = st.sidebar.selectbox("Colonne cible (classe)", options=all_cols, index=all_cols.index(default_target))

discretize_numeric_target = st.sidebar.checkbox(
    "Discrétiser la cible si numérique", value=False,
    help="Transforme une cible numérique continue en classes (bins) pour rester en classification."
)
nb_bins = st.sidebar.slider("Nombre de classes (bins)", 2, 7, 3, 1)
binning_method = st.sidebar.selectbox("Méthode de binning", ["quantiles", "largeur égale"], index=0)

if target_col not in df.columns:
    st.error("La colonne cible sélectionnée n’existe pas.")
    st.stop()

y_raw = df[target_col]
X = df.drop(columns=[target_col])

# Anti-fuite rapide
if y_raw.name in X.columns:
    st.error("⚠️ Data leakage : la cible est présente dans X.")
    st.stop()

# Discrétisation si besoin
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

num_cols, cat_cols = infer_types(df, target_col)
class_counts = y.value_counts()

# Avertissement si cible continue non discrétisée avec trop de valeurs distinctes
if pd.api.types.is_numeric_dtype(y) and not discretize_numeric_target and y.nunique() > 20:
    st.sidebar.warning(
        f"La cible '{target_col}' a {y.nunique()} valeurs distinctes (continue). "
        "Active la discrétisation pour rester en classification."
    )


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(
    ["🔎 Traitement des données", "📊 Visualisations", "🧠 Modélisation", "📈 Évaluation"]
)

# === TAB 1 — Traitement =======================================================
with tabs_1:
    st.subheader("Exploration & Préparation")

    st.markdown("### Aperçu")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Échantillons", len(df))
    c2.metric("Variables totales", df.shape[1])
    c3.metric("Numériques", len(num_cols))
    c4.metric("Catégorielles", len(cat_cols))

    st.markdown("### Statistiques descriptives")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("### Valeurs manquantes")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].to_frame("Nb_NA")
    if miss_df.empty:
        st.success("Aucune valeur manquante détectée !")
    else:
        st.warning("Des valeurs manquantes existent.")
        st.dataframe(miss_df)

    st.markdown("### Réglages de préparation")
    with st.expander("Imputation, normalisation, outliers, sélection de variables"):
        cA, cB = st.columns(2)
        scaling = cA.selectbox("Normalisation", ["Aucun", "StandardScaler", "MinMaxScaler", "RobustScaler"], index=1)
        impute_num = cA.selectbox("Imputation numériques", ["mean", "median", "most_frequent"], index=1)
        impute_cat = cA.selectbox("Imputation catégorielles", ["most_frequent", "constant"], index=0)

        outlier_mode = cB.selectbox("Gestion des outliers (num.)", ["Aucun", "Winsorisation (IQR)"], index=1)
        outlier_iqr_k = cB.slider("Seuil IQR (k)", 0.5, 3.0, 1.5, 0.1)

        sel_method = cB.selectbox("Sélection de variables", ["Aucune", "VarianceThreshold", "SelectKBest (ANOVA)"], index=0)
        k_best = cB.slider(
            "k (si SelectKBest)",
            min_value=1,
            max_value=max(1, len(num_cols) + len(cat_cols)),
            value=min(10, max(1, len(num_cols))),
        )

    st.session_state["prep_params"] = dict(
        scaling=scaling,
        impute_num=impute_num,
        impute_cat=impute_cat,
        outlier_mode=outlier_mode,
        outlier_iqr_k=outlier_iqr_k,
        sel_method=sel_method,
        k_best=k_best,
    )

    # Avertissement classes rares
    rare = class_counts[class_counts < 2]
    if not rare.empty:
        st.error(
            "⚠️ Classe(s) ultra-rare(s) détectée(s) (moins de 2 échantillons) : "
            + truncate_rare_list(rare)
            + ". Le split stratifié et la CV peuvent échouer."
        )

# === TAB 2 — Visualisations ===================================================
with tabs_2:
    st.subheader("Visualisations interactives")

    st.markdown("#### Distribution d’une variable")
    col_to_plot = st.selectbox("Variable numérique à visualiser", options=num_cols if num_cols else [None])
    if col_to_plot:
        hist = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{col_to_plot}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
                color=alt.Color(f"{target_col}:N") if pd.api.types.is_object_dtype(y) or y.dtype.name == "category" else alt.value(None),
                tooltip=[col_to_plot, target_col],
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Aucune variable numérique détectée.")

    st.markdown("#### Matrice de corrélation (numériques)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        corr_df = corr.reset_index().melt(id_vars="index")
        corr_df.columns = ["Var1", "Var2", "Corr"]
        heat = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(x="Var2:N", y="Var1:N", color=alt.Color("Corr:Q"), tooltip=["Var1", "Var2", "Corr"])
            .properties(height=400)
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques pour une corrélation.")

    st.markdown("#### Scatter matrix (pairplot-like)")
    if len(num_cols) >= 2:
        cols_select = st.multiselect("Colonnes numériques à inclure", num_cols, default=num_cols[:4])
        if len(cols_select) >= 2:
            sm = make_scatter_matrix(df[[*cols_select, target_col]], cols_select, color_by=target_col)
            st.altair_chart(sm, use_container_width=True)
        else:
            st.info("Sélectionne au moins 2 colonnes.")
    else:
        st.info("Pas assez de colonnes numériques pour un scatter matrix.")

    st.markdown("#### Répartition des classes")
    class_df = class_counts.reset_index()
    class_df.columns = ["Classe", "Nombre"]
    bar = alt.Chart(class_df).mark_bar().encode(
        x=alt.X("Classe:N", sort="-y"),
        y=alt.Y("Nombre:Q"),
        tooltip=["Classe", "Nombre"]
    ).properties(height=300)
    st.altair_chart(bar, use_container_width=True)

# === TAB 3 — Modélisation =====================================================
with tabs_3:
    st.subheader("Entraînement de modèles")

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Taille du test (%)", 10, 40, 20, 5) / 100.0
    random_state = c2.number_input("Random state", 0, 9999, 42, 1)
    do_cv = c3.checkbox("Faire une validation croisée (jusqu’à 5-fold)", value=False)

    use_balanced = st.checkbox("Pondérer les classes (class_weight='balanced')", value=False)

    model_name = st.selectbox("Modèle", ["LogisticRegression", "RandomForest", "KNN", "SVM (RBF)"], index=1)

    with st.expander("Hyperparamètres"):
        if model_name == "LogisticRegression":
            C = st.slider("C (inverse régularisation)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("max_iter", 100, 2000, 500, 50)
            clf = LogisticRegression(
                C=C, max_iter=max_iter, multi_class="ovr",
                class_weight=("balanced" if use_balanced else None)
            )
        elif model_name == "RandomForest":
            n_estimators = st.slider("n_estimators", 50, 500, 200, 10)
            max_depth = st.slider("max_depth", 1, 50, 10, 1)
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                class_weight=("balanced" if use_balanced else None)
            )
        elif model_name == "KNN":
            n_neighbors = st.slider("n_neighbors", 1, 25, 5, 1)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            C = st.slider("C", 0.01, 10.0, 1.0, 0.01)
            gamma = st.selectbox("gamma", ["scale", "auto"], index=0)
            clf = SVC(
                C=C, gamma=gamma, probability=True,
                random_state=random_state,
                class_weight=("balanced" if use_balanced else None)
            )

    # Préprocesseur (depuis les réglages)
    prep = st.session_state.get("prep_params", {})
    preprocessor = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        scaling=prep.get("scaling", "StandardScaler"),
        impute_num=prep.get("impute_num", "median"),
        impute_cat=prep.get("impute_cat", "most_frequent"),
        outlier_mode=prep.get("outlier_mode", "Winsorisation (IQR)"),
        outlier_iqr_k=prep.get("outlier_iqr_k", 1.5),
    )

    # Sélection de variables optionnelle
    selector = None
    if prep.get("sel_method") == "VarianceThreshold":
        selector = VarianceThreshold(threshold=0.0)
    elif prep.get("sel_method") == "SelectKBest (ANOVA)":
        selector = SelectKBest(score_func=f_classif, k=prep.get("k_best", 10))

    steps = [("preprocess", preprocessor)]
    if selector is not None:
        steps.append(("feature_select", selector))
    steps.append(("model", clf))
    model = Pipeline(steps=steps)

    # Split robuste (stratify auto si possible)
    can_stratify = stratify_feasible(y, test_size)
    if not can_stratify:
        st.warning(
            "Stratification impossible : certaines classes sont trop petites "
            f"(min={int(class_counts.min())}). `stratify=y` sera désactivé.\n"
            "👉 Solutions : réduire la taille de test, rééquilibrer/supprimer la classe ultra-rare."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    # Répartition post-split
    def _counts(s: pd.Series) -> pd.Series:
        return pd.Series(s.value_counts().to_dict(), name="count")

    with st.expander("Répartition des classes après split"):
        dist = pd.DataFrame({"train": _counts(y_train), "test": _counts(y_test)}).fillna(0).astype(int)
        st.dataframe(dist)

    if st.button("Entraîner le modèle"):
        model.fit(X_train, y_train)
        st.success("Modèle entraîné !")

        # Sauvegarde session
        st.session_state["trained_model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["class_labels"] = sorted(y.unique())

        # Overfitting check (train vs test)
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
            st.warning(f"Possible sur-apprentissage : écart de {gap:.2f} entre train et test.")
        else:
            st.success("Pas d’overfitting significatif détecté !")

        # Scores rapides test
        c1, c2 = st.columns(2)
        c1.metric("Accuracy (test)", f"{test_acc:.3f}")
        c2.metric("F1 pondéré (test)", f"{test_f1:.3f}")

        # Validation croisée sûre
        if do_cv:
            min_per_class = int(class_counts.min())
            cv_folds = max(2, min(5, min_per_class))
            if cv_folds < 2:
                st.info("Validation croisée désactivée (classes trop petites pour CV).")
            else:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
                st.write(f"Validation croisée (accuracy, {cv_folds}-fold) : **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**")

        # Export du pipeline
        buf = io.BytesIO()
        pickle.dump(model, buf)
        st.download_button(
            label="💾 Télécharger le pipeline entraîné (.pkl)",
            data=buf.getvalue(),
            file_name=f"pipeline_{model_name}.pkl",
            mime="application/octet-stream",
        )

        # Importances / coefficients (alignées avec noms de features post-preprocess + sélection)
        with st.expander("Importance des variables / coefficients"):
            try:
                # Noms après preprocess
                try:
                    feat_names = list(model.named_steps["preprocess"].get_feature_names_out())
                except Exception:
                    feat_names = None

                # Appliquer masque du sélecteur si présent
                if "feature_select" in model.named_steps and hasattr(model.named_steps["feature_select"], "get_support"):
                    mask = model.named_steps["feature_select"].get_support()
                    if feat_names is not None:
                        feat_names = [n for n, keep in zip(feat_names, mask) if keep]

                if hasattr(model.named_steps["model"], "feature_importances_"):
                    imp = model.named_steps["model"].feature_importances_
                    if feat_names is None:
                        feat_names = [f"feat_{i}" for i in range(len(imp))]
                    imp_df = pd.DataFrame({"Feature": feat_names[: len(imp)], "Importance": imp}) \
                               .sort_values("Importance", ascending=False).head(25)
                    chart = alt.Chart(imp_df).mark_bar().encode(
                        x=alt.X("Importance:Q"), y=alt.Y("Feature:N", sort="-x")
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)

                elif hasattr(model.named_steps["model"], "coef_"):
                    coefs = model.named_steps["model"].coef_
                    coef_mean = np.mean(np.abs(coefs), axis=0)
                    if feat_names is None:
                        coef_df = pd.DataFrame({"Indice": range(len(coef_mean)), "Poids": coef_mean})
                        chart = alt.Chart(coef_df.sort_values("Poids", ascending=False).head(25)).mark_bar().encode(
                            x=alt.X("Poids:Q"), y=alt.Y("Indice:N", sort="-x")
                        ).properties(height=400)
                    else:
                        coef_df = pd.DataFrame({"Feature": feat_names[: len(coef_mean)], "Poids": coef_mean})
                        chart = alt.Chart(coef_df.sort_values("Poids", ascending=False).head(25)).mark_bar().encode(
                            x=alt.X("Poids:Q"), y=alt.Y("Feature:N", sort="-x")
                        ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Le modèle ne fournit pas d’importances/coefs exploitables.")
            except Exception as e:
                st.warning(f"Impossible d’afficher les importances: {e}")

# === TAB 4 — Évaluation =======================================================
with tabs_4:
    st.subheader("Évaluation du modèle")

    if "trained_model" not in st.session_state:
        st.info("Entraîne d’abord un modèle dans l’onglet *Modélisation*.")
    else:
        model = st.session_state["trained_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        labels = st.session_state["class_labels"]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model.named_steps["model"], "predict_proba") else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        c2.metric("Precision (weighted)", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
        c3.metric("Recall (weighted)", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
        c4.metric("F1 (weighted)", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")

        st.markdown("### Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
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