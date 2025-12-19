import os
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ======== THEME ALTAIR TRANSPARENT (fond = même que la page) ========
def _transparent_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {
                "fill": "transparent",
                "strokeWidth": 0
            }
        }
    }

alt.themes.register("transparent", _transparent_theme)
alt.themes.enable("transparent")
# ====================================================================

# ================== CHEMINS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models", "multitask_bertweet_epoch5")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
DATA_TEST_DIR = os.path.join(BASE_DIR, "data_test")

EMOTION_CSV = os.path.join(DATA_CLEAN_DIR, "emotion_clean.csv")
PROPAGANDA_CSV = os.path.join(DATA_CLEAN_DIR, "propaganda_clean.csv")
BIAS_CSV = os.path.join(DATA_CLEAN_DIR, "bias_clean.csv")

EMOTION_TEST_CSV = os.path.join(DATA_TEST_DIR, "emotion_test.csv")
PROPAGANDA_TEST_CSV = os.path.join(DATA_TEST_DIR, "propaganda_test.csv")  # ✅ corrigé
BIAS_TEST_CSV = os.path.join(DATA_TEST_DIR, "bias_test.csv")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== MODELE MULTITACHE ==================
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="vinai/bertweet-base", num_emotions=6, num_binary=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = 768
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        self.binary_head = nn.Linear(hidden_size, num_binary)

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]
        if task_type == "emotion":
            return self.emotion_head(cls_vec)
        elif task_type == "binary":
            return self.binary_head(cls_vec)
        else:
            raise ValueError("task_type must be 'emotion' or 'binary'")

# ================== CONFIG UI ==================
st.set_page_config(
    page_title="Emotion & Manipulation – Dashboard & Prédictions",
    layout="wide",
    page_icon="🎭",
)

st.markdown(
    """
    <style>
    :root {
        --bg-main: #f4f7ff;
        --bg-sidebar: #2f3e5e;
        --sidebar-text: #ecf0f1;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        color: var(--sidebar-text);
    }

    section[data-testid="stSidebar"] h1 {
        color: var(--sidebar-text);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.6rem;
        font-size: 16px;
        color: var(--sidebar-text);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--sidebar-text);
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .big-intro {
        font-size: 20px;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    header[data-testid="stHeader"] {
        background-color: var(--bg-main) !important;
        padding-top: 8px;
    }

    header[data-testid="stHeader"]::before {
        background-color: transparent !important;
    }

    #root header button,
    #root header svg {
        color: #333 !important;
    }
    
    /* ===== Cartes KPIs du dashboard ===== */
    .kpi-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 14px 18px;
        border: 1px solid #dde3ff;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
        margin-top: 0.5rem;
    }
    .kpi-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        text-align: center;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        text-align: center;
    }
    .kpi-card.kpi-1 { border-top: 4px solid #4C6FFF; }
    .kpi-card.kpi-2 { border-top: 4px solid #FF9F80; }
    .kpi-card.kpi-3 { border-top: 4px solid #1ABC9C; }
    .kpi-card.kpi-4 { border-top: 4px solid #F97316; }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== CACHES ==================
@st.cache_resource
def load_model_and_tokenizer():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    backbone_name = checkpoint["backbone_name"]
    num_emotions = checkpoint["num_emotions"]
    num_binary = checkpoint["num_binary"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    model = MultiTaskBERT(
        model_name=backbone_name,
        num_emotions=num_emotions,
        num_binary=num_binary
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_to_emotion = checkpoint["label_to_emotion"]
    class_weights = torch.tensor(checkpoint["class_weights"], dtype=torch.float32).to(device)

    return model, tokenizer, label_to_emotion, class_weights


@st.cache_data
def load_clean_datasets():
    dfs = {}
    if os.path.exists(EMOTION_CSV):
        dfs["emotion"] = pd.read_csv(EMOTION_CSV)
    if os.path.exists(PROPAGANDA_CSV):
        dfs["propaganda"] = pd.read_csv(PROPAGANDA_CSV)
    if os.path.exists(BIAS_CSV):
        dfs["bias"] = pd.read_csv(BIAS_CSV)
    return dfs


@st.cache_data
def load_test_datasets():
    dfs = {}
    if os.path.exists(EMOTION_TEST_CSV):
        dfs["emotion_test"] = pd.read_csv(EMOTION_TEST_CSV)
    if os.path.exists(PROPAGANDA_TEST_CSV):
        dfs["propaganda_test"] = pd.read_csv(PROPAGANDA_TEST_CSV)
    if os.path.exists(BIAS_TEST_CSV):
        dfs["bias_test"] = pd.read_csv(BIAS_TEST_CSV)
    return dfs

# ================== LOGIQUE PREDICTION ==================
binary_label_to_name = {
    0: "Non manipulateur / non biaisé",
    1: "Manipulateur / biaisé"
}

EMOTION_COLORS = {
    "Sadness": "#4C6FFF",
    "Joy": "#FF9F80",
    "Love": "#FF6FB5",
    "Anger": "#FF4B4B",
    "Fear": "#9B59B6",
    "Surprise": "#1ABC9C",
    "Neutral": "#95A5A6",
}

MANIP_COLORS = {
    "Non manipulateur / non biaisé": "#2ECC71",
    "Manipulateur / biaisé": "#3498DB",
}

# ===== Couleurs pour le badge du nom de dataset =====
DATASET_BADGE_COLORS = {
    "emotion": ("#EEF2FF", "#4C6FFF"),
    "propaganda": ("#FFF7ED", "#F97316"),
    "bias": ("#ECFEFF", "#0891B2"),
}

# ⚠️ Seuils utilisés dans la logique hiérarchique
EMOTION_THRESHOLD = 0.5      # intensité max minimale pour dire "émotionnel"
MANIP_THRESHOLD = 0.85       # seuil strict pour dire "manipulateur"


def predict_text(text, model, tokenizer, label_to_emotion, device, max_length=128):
    model.eval()
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        # Tête émotion
        logits_e = model(input_ids, attention_mask, task_type="emotion")
        probs_e = torch.softmax(logits_e, dim=-1)[0]
        pred_e = probs_e.argmax().item()
        emotion_name = label_to_emotion[pred_e]

        # Tête manipulation / biais
        logits_b = model(input_ids, attention_mask, task_type="binary")
        probs_b = torch.softmax(logits_b, dim=-1)[0]
        pred_b = probs_b.argmax().item()
        binary_name = binary_label_to_name[pred_b]

    return {
        "text": text,
        "emotion_label": pred_e,
        "emotion_name": emotion_name,
        "emotion_probs": probs_e.cpu().tolist(),
        "binary_label": pred_b,
        "binary_name": binary_name,
        "binary_probs": probs_b.cpu().tolist(),
    }


def classify_text_final(
    text,
    model,
    tokenizer,
    label_to_emotion,
    emotion_th: float,
    manip_th: float,
):
    """
    Logique finale :
    1) On prédit émotions + manipulation.
    2) On décide d'abord si le texte est neutre ou émotionnel.
    3) On raffine la décision "manipulateur / non" avec une petite heuristique
       basée sur des motifs typiques de propagande (us vs them, appel à l'action, urgence...).
    """

    base_pred = predict_text(text, model, tokenizer, label_to_emotion, device)
    emo_probs = base_pred["emotion_probs"]
    manip_probs = base_pred["binary_probs"]

    emo_max = max(emo_probs)
    emo_name = base_pred["emotion_name"]
    p_non_manip, p_manip = manip_probs[0], manip_probs[1]

    # --------------------------
    # 1. Détection de patterns de propagande dans le texte
    # --------------------------
    t = text.lower()

    propaganda_keywords = [
        "wake up", "last chance", "stand up", "join us", "join the movement",
        "save our", "save your", "fight for", "fight against",
        "they are lying", "they are liars","doing with our",
        "they want to control", "control your mind",
        "they control everything", "they control your life",
        "defend our values", "true patriots", "real patriots",
        "before it's too late", "before it is too late",
        "if you don’t act now", "if you don't act now",
        "your children will pay", "they will destroy everything",
        "enemies of progress", "resistance", "reclaim our future",
    ]

    group_pronouns = ["they", "them", "their", "our", "us", "we", "you people"]
    call_to_action = ["stand up", "join", "share this", "act now", "rise up"]

    has_propaganda_kw = any(kw in t for kw in propaganda_keywords)
    has_group_pron = any(p in t for p in group_pronouns)
    has_call_to_action = any(c in t for c in call_to_action)

    is_propaganda_style = has_propaganda_kw or (has_group_pron and has_call_to_action)

    # --------------------------
    # 2. Décision principale
    # --------------------------

    # Cas 1 : pas d'émotion dominante -> neutre, SAUF si texte typé propagande + manip très forte
    if emo_max < emotion_th:
        if is_propaganda_style and p_manip > 0.80:
            final_label = "emotionnelle_manipulatrice"
            final_emotion = None  # émotion peu claire
            final_description = (
                "Texte à tonalité de propagande / appel à l'action, "
                "avec une forte probabilité de manipulation, "
                "mais sans émotion dominante très claire."
            )
            color = "red"
        else:
            final_label = "neutre"
            final_description = "Texte neutre (aucune émotion dominante détectée)."
            final_emotion = None
            color = "gray"

    else:
        # Cas 2 : émotion détectée -> on regarde la manipulation
        final_emotion = emo_name

        # 2.a texte clairement manipulatoire
        if p_manip > manip_th and (is_propaganda_style or p_manip > 0.90):
            final_label = "emotionnelle_manipulatrice"
            final_description = (
                f"Texte émotionnel et **manipulateur**, émotion dominante : **{final_emotion}**."
            )
            color = "red"
        # 2.b émotion présente mais manipulation faible
        elif p_manip < 0.40 and not is_propaganda_style:
            final_label = "emotionnelle_non_manipulatrice"
            final_description = (
                f"Texte émotionnel **non manipulateur**, émotion dominante : **{final_emotion}**."
            )
            color = "green"
        # 2.c zone grise -> on reste prudent
        else:
            final_label = "emotionnelle_non_manipulatrice"
            final_description = (
                f"Texte émotionnel avec des indices possibles de manipulation "
                f"(émotion dominante : **{final_emotion}**), mais confiance limitée."
            )
            color = "green"

    base_pred["final_label"] = final_label
    base_pred["final_description"] = final_description
    base_pred["final_emotion"] = final_emotion
    base_pred["color"] = color
    return base_pred



def batch_predict_emotion(texts, model, tokenizer, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask, task_type="emotion")
            batch_pred = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
        preds.extend(batch_pred.tolist())
    return np.array(preds)


def batch_predict_binary(texts, model, tokenizer, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask, task_type="binary")
            batch_pred = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
        preds.extend(batch_pred.tolist())
    return np.array(preds)

# ================== CHARGEMENT ==================
model, tokenizer, label_to_emotion, class_weights_tensor = load_model_and_tokenizer()
dfs = load_clean_datasets()
test_dfs = load_test_datasets()

if "pred_history" not in st.session_state:
    st.session_state["pred_history"] = []

# ================== SIDEBAR ==================
st.sidebar.title("🧭 Explorer")
page = st.sidebar.radio(
    "",
    ["dashboard", "pred", "about", "eval"],
    format_func=lambda x: {
        "dashboard": "📊 Dashboard",
        "pred": "🔮 Prédictions",
        "about": "ℹ️ À propos",
        "eval": "📏 Évaluation",
    }[x]
)

# ================== PAGE 1 : DASHBOARD ==================
if page == "dashboard":
    st.markdown('<div class="main-title">📊 Dashboard des datasets nettoyés</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-subtitle">
        Cette page permet de :
        <ul>
            <li>visualiser les jeux de données propres (<b>emotion</b>, <b>propaganda</b>, <b>bias</b>),</li>
            <li>explorer la distribution des classes,</li>
            <li>inspecter quelques exemples de textes par classe,</li>
            <li>et télécharger les données au format CSV.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not dfs:
        st.error("Aucun dataset trouvé. Vérifie le dossier `data_clean`.")
    else:
        st.markdown("📁 **Choisir un dataset à explorer :**")
        dataset_choice = st.selectbox("", list(dfs.keys()))

        df = dfs[dataset_choice]

        # ------ Titre + badge dataset ------
        bg_badge, txt_badge = DATASET_BADGE_COLORS.get(
            dataset_choice,
            ("#E5E7EB", "#111827")
        )

        st.markdown(
            f"""
            <div style="
                margin-top: 1.5rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                font-size: 22px;
            ">
                <span style="font-weight: 700; color: #111827;">
                    Dataset :
                </span>
                <span style="
                    padding: 4px 18px;
                    border-radius: 999px;
                    background: {bg_badge};
                    color: {txt_badge};
                    font-weight: 700;
                    border: 1px solid rgba(148,163,184,0.8);
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                    font-size: 13px;
                ">
                    {dataset_choice}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"**Nombre de lignes :** {len(df):,}".replace(",", " "))

        # ==================== CARTES KPIs ====================
        st.markdown("### 🧠 Vue globale du dataset")

        total_texts = len(df)
        avg_len = df["text_len"].mean() if "text_len" in df.columns else None
        n_classes = df["label"].nunique() if "label" in df.columns else None

        if "label" in df.columns:
            top_class_raw = df["label"].value_counts().idxmax()
            if dataset_choice == "emotion" and isinstance(label_to_emotion, dict):
                top_class = label_to_emotion[int(top_class_raw)]
            else:
                top_class = str(top_class_raw)
        else:
            top_class = None

        def fmt_int(x):
            return f"{x:,}".replace(",", " ") if x is not None else "—"

        def fmt_float(x):
            return f"{x:.1f}" if x is not None else "—"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="kpi-card kpi-1">
                    <div class="kpi-label">Nombre total de textes</div>
                    <div class="kpi-value">{fmt_int(total_texts)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="kpi-card kpi-2">
                    <div class="kpi-label">Longueur moyenne des textes</div>
                    <div class="kpi-value">{fmt_float(avg_len)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="kpi-card kpi-3">
                    <div class="kpi-label">Nombre de classes</div>
                    <div class="kpi-value">{fmt_int(n_classes)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="kpi-card kpi-4">
                    <div class="kpi-label">Classe majoritaire</div>
                    <div class="kpi-value">{top_class if top_class is not None else "—"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # ================== FIN CARTES KPIs ====================

        tab1, tab2, tab3 = st.tabs(["🧾 Aperçu", "📊 Classes", "🔎 Exemples"])

        with tab1:
            st.markdown("#### Aperçu des données nettoyées")
            st.dataframe(df.head(15))
            if "text_len" in df.columns:
                st.markdown("#### Statistiques sur la longueur des textes")
                st.write(df["text_len"].describe()[["mean", "50%", "75%", "max"]])
                hist = df["text_len"].clip(0, 300)
                counts = hist.value_counts().sort_index()
                hist_df = pd.DataFrame(
                    {"longueur": counts.index, "count": counts.values}
                ).set_index("longueur")
                st.bar_chart(hist_df)

        with tab2:
            if "label" in df.columns:
                st.markdown("#### Distribution des classes")
                label_counts = df["label"].value_counts().sort_index()

                if dataset_choice == "emotion" and isinstance(label_to_emotion, dict):
                    readable_labels = [label_to_emotion[int(i)] for i in label_counts.index]
                    df_plot = pd.DataFrame({
                        "Émotion": readable_labels,
                        "Nombre": label_counts.values
                    })

                    chart = (
                        alt.Chart(df_plot)
                        .mark_bar()
                        .encode(
                            x=alt.X("Émotion:N", sort=None, title="Émotion"),
                            y=alt.Y("Nombre:Q", title="Nombre de textes"),
                            color=alt.Color(
                                "Émotion:N",
                                scale=alt.Scale(
                                    domain=list(EMOTION_COLORS.keys()),
                                    range=list(EMOTION_COLORS.values())
                                ),
                                legend=None
                            ),
                            tooltip=["Émotion", "Nombre"]
                        )
                        .properties(height=300)
                        .configure_view(fill="transparent")
                        .configure_axis(
                            gridColor="#d3dae8",
                            labelColor="#22303C",
                            titleColor="#22303C"
                        )
                    )

                    st.altair_chart(chart, use_container_width=True)
                    st.write(df_plot.set_index("Émotion"))

                elif dataset_choice in ["propaganda", "bias"] and set(label_counts.index.tolist()) <= {0, 1}:
                    readable_labels = [binary_label_to_name[int(i)] for i in label_counts.index]
                    df_plot = pd.DataFrame({
                        "Classe": readable_labels,
                        "Nombre": label_counts.values
                    })

                    chart = (
                        alt.Chart(df_plot)
                        .mark_bar()
                        .encode(
                            x=alt.X("Classe:N", sort=None, title="Classe"),
                            y=alt.Y("Nombre:Q", title="Nombre de textes"),
                            color=alt.Color(
                                "Classe:N",
                                scale=alt.Scale(
                                    domain=list(MANIP_COLORS.keys()),
                                    range=list(MANIP_COLORS.values())
                                ),
                                legend=None
                            ),
                            tooltip=["Classe", "Nombre"]
                        )
                        .properties(height=300)
                        .configure_view(fill="transparent")
                        .configure_axis(
                            gridColor="#d3dae8",
                            labelColor="#22303C",
                            titleColor="#22303C"
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                    st.write(df_plot.set_index("Classe"))

                else:
                    readable_labels = label_counts.index.astype(str)
                    label_df = pd.DataFrame({
                        "Classe": readable_labels,
                        "Nombre": label_counts.values
                    }).set_index("Classe")
                    st.bar_chart(label_df)
                    st.write(label_df)
            else:
                st.info("Aucune colonne `label` trouvée dans ce dataset.")

        with tab3:
            if "label" in df.columns and "clean_text" in df.columns:
                st.markdown("#### Exemples de textes par classe")
                unique_labels = sorted(df["label"].unique())
                if dataset_choice == "emotion" and isinstance(label_to_emotion, dict):
                    label_display = [f"{lbl} – {label_to_emotion[int(lbl)]}" for lbl in unique_labels]
                    chosen = st.selectbox("Choisir un label :", label_display)
                    class_selected = int(chosen.split("–")[0].strip())
                else:
                    class_selected = st.selectbox(
                        "Choisir un label pour voir des exemples :", unique_labels
                    )
                sample_rows = df[df["label"] == class_selected].sample(
                    n=min(10, (df["label"] == class_selected).sum()), random_state=0
                )
                for _, row in sample_rows.iterrows():
                    if dataset_choice == "emotion" and isinstance(label_to_emotion, dict):
                        emo_name = label_to_emotion[int(row["label"])]
                        st.markdown(f"- **[{emo_name}]** `{row['clean_text']}`")
                    else:
                        st.markdown(f"- `{row['clean_text']}`")
            else:
                st.info("Pas de colonnes compatibles (`label`, `clean_text`).")

        # ===== Bouton de téléchargement en bas de page =====
        st.markdown(
            """
            <div style="
                margin-top: 40px;
                margin-bottom: 10px;
                text-align: center;
                font-size: 20px;
                font-weight: 600;
            ">
                📥 Télécharger ce dataset
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Télécharger ce dataset (CSV)",
            data=csv_bytes,
            file_name=f"{dataset_choice}_clean.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ================== PAGE 2 : PREDICTION ==================
elif page == "pred":
    st.markdown('<div class="main-title">🔮 Prédiction Emotion & Manipulation</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="big-intro">
        Entrez un texte (tweet, message, etc.) et le modèle prédira :
        <ul>
            <li>si le texte est <b>neutre</b>,</li>
            <li>ou <b>émotionnel non manipulateur</b>,</li>
            <li>ou <b>émotionnel manipulateur</b>,</li>
        </ul>
        tout en indiquant l'<b>émotion dominante</b> et les <b>probabilités complètes</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    example_texts = [
        "I feel so sad and empty today...",
        "You are being lied to, wake up, they control everything.",
        "The meeting is scheduled at 10 AM tomorrow in the main conference room.",
        "I feel so sad and empty today, nothing seems to matter anymore.",
        "Please send me the report before Friday so we can finalize the project.",
        "This is unacceptable! I'm really angry about how they treated us.",
        "I'm really scared about the future, everything feels so uncertain.",
        "If you stay silent today, tomorrow your rights will be gone forever.",
        "If you don’t rise up today, your family will regret it forever.",
        "Honestly, this whole situation makes me furious. Nothing seems fair anymore.",
        "I really tried to fix things, but everything feels hopeless right now.",
        "The latest report will be uploaded once the revision is complete.",
        "The file you requested is available in your shared drive.",
        "I feel so grateful today, everything seems to finally work out for me!",
        "I don’t know why, but I feel so empty today, nothing can cheer me up.",
        "I'm extremely upset about how they handled this situation. It's just unfair.",
        "i love how grateful to him i feel each day for his love kindness and tender mercies",
        "The temperature will range between 15 and 18 degrees today with light rain in the evening.",
        "Please submit your application before the deadline mentioned on the website.",
        "I feel so sad and empty today. Nothing seems to make sense anymore.",
        "I’m really scared about the exam tomorrow, I can’t stop thinking about failing.",
        "I’m so angry about how they treated me at work today. It was completely unfair.",
        "This is our last chance to save our country. If you stay silent now, you will lose all your rights forever.",
        "True patriots know the truth. Only we are brave enough to fight for real freedom. Join our movement today.",
        "Please upload the report before Friday afternoon.",
        "Your file is now available in the shared drive.",
        "The latest version of the document has been approved and archived.",
        "I'm so sad and empty today, nothing seems to matter anymore.",
        "I'm really scared about tomorrow's exam, I hope I won't fail.",
        "I'm so angry about how they treated me at work, it's extremely unfair.",
        "I feel truly blessed, thank you for always being there for me.",
        "This is your last chance to save our country. Stand up now or lose everything.",
        "If you don't act today, your children will pay the price tomorrow.",
        "They want to control your life, your job, your future. Open your eyes.",
        "Real patriots must join us now. Together we fight for true freedom.",
        "True patriots know the truth. Join us before it's too late.",
        "They will destroy everything unless we stop them now.",
        "Stand up and defend our values before they disappear.",
        "This movement is the only way to reclaim our future. Join now."

    ]

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    def on_example_change():
        choice = st.session_state["example_choice"]
        if choice != "(aucun)":
            st.session_state["input_text"] = choice

    st.markdown("📌 **Choisir ou saisir un exemple :**")
    user_text = st.text_area(
        "",
        value=st.session_state["input_text"],
        height=150,
        key="input_text"
    )

    st.selectbox(
        "Insérer un exemple prédéfini (optionnel) :",
        ["(aucun)"] + example_texts,
        key="example_choice",
        on_change=on_example_change
    )

    if st.button("Analyser le texte"):
        if not user_text.strip():
            st.warning("Veuillez entrer ou saisir un texte dans le champ ci-dessus.")
        else:
            with st.spinner("Prédiction en cours..."):
                pred = classify_text_final(
                    user_text,
                    model,
                    tokenizer,
                    label_to_emotion,
                    emotion_th=EMOTION_THRESHOLD,
                    manip_th=MANIP_THRESHOLD
                )

            st.subheader("🧾 Résultat global")

            label_human = {
                "neutre": "Texte neutre",
                "emotionnelle_non_manipulatrice": "Texte émotionnel non manipulateur",
                "emotionnelle_manipulatrice": "Texte émotionnel manipulateur"
            }[pred["final_label"]]

            color = pred["color"]
            if color == "red":
                bg = "#ffdddd"; border = "#ff8888"
            elif color == "green":
                bg = "#ddffdd"; border = "#88cc88"
            else:
                bg = "#eeeeee"; border = "#bbbbbb"

            st.markdown(
                f"""
                <div style='padding:16px;border-radius:12px;background-color:{bg};
                border:1px solid {border};font-size:18px;'>
                  <div style='font-size:20px;font-weight:700;margin-bottom:4px;'>
                    {label_human}
                  </div>
                  <div>{pred['final_description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🎭 Probabilités & ton manipulateur")

            emo_names = [label_to_emotion[i] for i in range(len(pred["emotion_probs"]))]
            emo_df = pd.DataFrame({"Emotion": emo_names, "Proba": pred["emotion_probs"]})

            bin_names = ["Non manipulateur / non biaisé", "Manipulateur / biaisé"]
            bin_df = pd.DataFrame({"Type": bin_names, "Proba": pred["binary_probs"]})

            chart_emo = (
                alt.Chart(emo_df)
                .mark_bar()
                .encode(
                    x=alt.X("Emotion:N", sort=None, title="Émotion"),
                    y=alt.Y("Proba:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilité"),
                    color=alt.Color(
                        "Emotion:N",
                        scale=alt.Scale(
                            domain=list(EMOTION_COLORS.keys()),
                            range=list(EMOTION_COLORS.values())
                        ),
                        legend=None
                    ),
                    tooltip=["Emotion", alt.Tooltip("Proba:Q", format=".2f")]
                )
                .properties(width=450, height=260, title="Probabilités des émotions")
            )

            chart_bin = (
                alt.Chart(bin_df)
                .mark_bar()
                .encode(
                    x=alt.X("Type:N", sort=None, title="Classe"),
                    y=alt.Y("Proba:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilité"),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(
                            domain=list(MANIP_COLORS.keys()),
                            range=list(MANIP_COLORS.values())
                        ),
                        legend=None
                    ),
                    tooltip=["Type", alt.Tooltip("Proba:Q", format=".2f")]
                )
                .properties(width=450, height=260, title="Probabilité de ton manipulateur")
            )

            combined = (
                alt.hconcat(chart_emo, chart_bin)
                .resolve_scale(y="shared")
                .configure_view(fill="transparent")
                .configure_axis(
                    gridColor="#d3dae8",
                    labelColor="#22303C",
                    titleColor="#22303C",
                )
            )

            st.altair_chart(combined, use_container_width=True)

            st.markdown("### 🔎 Détails")
            st.write("**Texte analysé :**")
            st.info(user_text)

            st.session_state["pred_history"].append({
                "Texte": user_text,
                "Label global": label_human,
                "Émotion dominante": pred["final_emotion"],
                "Proba manipulatrice": round(pred["binary_probs"][1], 3)
            })

    if st.session_state["pred_history"]:
        with st.expander("📚 Historique des prédictions"):
            hist_df = pd.DataFrame(st.session_state["pred_history"])
            st.dataframe(hist_df, use_container_width=True)

# ================== PAGE 3 : À PROPOS ==================
elif page == "about":
    st.markdown('<div class="main-title">ℹ️ À propos du modèle</div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Architecture")
    st.markdown(
        """
        - **Backbone** : **vinai/bertweet-base**  
        - **Modèle multitâche** avec deux têtes de classification :  
          - Émotions (6 classes : Sadness, Joy, Love, Anger, Fear, Surprise)  
          - Manipulation / biais (binaire : non manipulateur vs manipulateur)  
        - Chaque tête est une couche linéaire appliquée au vecteur [CLS].
        """
    )

    st.markdown("### 📚 Jeux de données utilisés")

    if dfs:
        cols = st.columns(3)
        with cols[0]:
            if "emotion" in dfs:
                total = len(dfs["emotion"])
                train = int(0.8 * total)
                val = int(0.1 * total)
                test = total - train - val
                st.markdown("**Emotion**")
                st.write(f"Total : {total:,}".replace(",", " "))
                st.write(f"Train ~ {train:,}".replace(",", " "))
                st.write(f"Validation ~ {val:,}".replace(",", " "))
                st.write(f"Test ~ {test:,}".replace(",", " "))

        with cols[1]:
            if "propaganda" in dfs:
                total = len(dfs["propaganda"])
                train = int(0.8 * total)
                val = int(0.1 * total)
                test = total - train - val
                st.markdown("**Propaganda**")
                st.write(f"Total : {total:,}".replace(",", " "))
                st.write(f"Train ~ {train:,}".replace(",", " "))
                st.write(f"Validation ~ {val:,}".replace(",", " "))
                st.write(f"Test ~ {test:,}".replace(",", " "))

        with cols[2]:
            if "bias" in dfs:
                total = len(dfs["bias"])
                train = int(0.8 * total)
                val = int(0.1 * total)
                test = total - train - val
                st.markdown("**News Bias**")
                st.write(f"Total : {total:,}".replace(",", " "))
                st.write(f"Train ~ {train:,}".replace(",", " "))
                st.write(f"Validation ~ {val:,}".replace(",", " "))
                st.write(f"Test ~ {test:,}".replace(",", " "))
    else:
        st.info("Les fichiers `data_clean` ne sont pas chargés.")

    st.markdown("### 📊 Performances de validation (résumé)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tâche émotion**")
        st.markdown(
            """
            - Accuracy validation ≈ **0.965**  
            - Loss validation ≈ **0.07**  
            - F1 (macro) : à confirmer sur le jeu de test.  
            """
        )
    with col2:
        st.markdown("**Tâche manipulation / biais**")
        st.markdown(
            """
            - Accuracy validation ≈ **0.86**  
            - F1 (macro) : à confirmer sur le jeu de test.  
            """
        )

    st.markdown("### ⚙️ Détails d'entraînement")
    st.markdown(
        """
        - Optimisation multitâche : **loss_totale = loss_émotion + loss_binaire**  
        - Poids de classes utilisés pour l’émotion (déséquilibre des classes)  
        - Entraînement sur **5 epochs**.  
        """
    )

# ================== PAGE 4 : ÉVALUATION ==================
elif page == "eval":
    st.markdown('<div class="main-title">📏 Évaluation sur les jeux de test</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Cette page évalue le modèle sur les **jeux de test officiels** :  
        **emotion_test.csv**, **propaganda_test.csv**, **bias_test.csv** .  
        Chaque fichier doit contenir :
        - une colonne **label** (entiers),  
        - une colonne **clean_text** ou **text** pour le contenu.
        """
    )

    if not test_dfs:
        st.warning("Aucun fichier de test trouvé dans `data_test`.")
    else:
        dataset_choice = st.selectbox(
            "Choisir un dataset de test à évaluer :",
            list(test_dfs.keys()),
            format_func=lambda x: {
                "emotion_test": "Emotion – test",
                "propaganda_test": "Propaganda – test",
                "bias_test": "Bias – test"
            }.get(x, x)
        )

        df_test = test_dfs[dataset_choice]
        st.markdown("#### Aperçu du jeu de test")
        st.dataframe(df_test.head())

        if "label" not in df_test.columns:
            st.error("Le fichier de test doit contenir une colonne `label`.")
        else:
            if "clean_text" in df_test.columns:
                text_col = "clean_text"
            elif "text" in df_test.columns:
                text_col = "text"
            else:
                st.error("Le fichier doit contenir `clean_text` ou `text`.")
                text_col = None

            if text_col is not None and st.button("🚀 Lancer l'évaluation sur ce dataset"):
                texts = df_test[text_col].astype(str).tolist()
                y_true = df_test["label"].to_numpy()
                st.info(f"Nombre d'exemples dans le test : {len(y_true)}")

                with st.spinner("Prédictions du modèle en cours..."):
                    if dataset_choice == "emotion_test":
                        y_pred = batch_predict_emotion(texts, model, tokenizer)
                        acc = accuracy_score(y_true, y_pred)
                        f1_macro = f1_score(y_true, y_pred, average="macro")
                        f1_weighted = f1_score(y_true, y_pred, average="weighted")

                        st.markdown("### 📈 Résultats – tâche émotion")
                        st.write(f"- Accuracy : **{acc:.3f}**")
                        st.write(f"- F1 macro : **{f1_macro:.3f}**")
                        st.write(f"- F1 weighted : **{f1_weighted:.3f}**")

                        labels_emo = sorted(label_to_emotion.keys())
                        cm = confusion_matrix(y_true, y_pred, labels=labels_emo)
                        emo_names = [label_to_emotion[i] for i in labels_emo]

                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"Vrai {n}" for n in emo_names],
                            columns=[f"Prédit {n}" for n in emo_names]
                        ).reset_index().melt(id_vars="index", var_name="Pred", value_name="Count")
                        cm_df.rename(columns={"index": "True"}, inplace=True)

                        chart_cm = (
                            alt.Chart(cm_df)
                            .mark_rect()
                            .encode(
                                x=alt.X("Pred:N", axis=alt.Axis(title="Prédit", labelAngle=0)),
                                y=alt.Y("True:N", axis=alt.Axis(title="Vrai")),
                                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                                tooltip=["True", "Pred", "Count"]
                            )
                            .properties(title="Matrice de confusion – émotions (test)", width=350, height=350)
                        )
                        st.altair_chart(chart_cm, use_container_width=True)

                    else:
                        y_pred = batch_predict_binary(texts, model, tokenizer)
                        acc = accuracy_score(y_true, y_pred)
                        f1_macro = f1_score(y_true, y_pred, average="macro")
                        f1_weighted = f1_score(y_true, y_pred, average="weighted")

                        st.markdown("### 📈 Résultats – tâche binaire (manipulation / biais)")
                        st.write(f"- Accuracy : **{acc:.3f}**")
                        st.write(f"- F1 macro : **{f1_macro:.3f}**")
                        st.write(f"- F1 weighted : **{f1_weighted:.3f}**")

                        labels_bin = [0, 1]
                        cm = confusion_matrix(y_true, y_pred, labels=labels_bin)
                        bin_names = ["Non manip.", "Manip."]

                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"Vrai {n}" for n in bin_names],
                            columns=[f"Prédit {n}" for n in bin_names]
                        ).reset_index().melt(id_vars="index", var_name="Pred", value_name="Count")
                        cm_df.rename(columns={"index": "True"}, inplace=True)

                        chart_cm = (
                            alt.Chart(cm_df)
                            .mark_rect()
                            .encode(
                                x=alt.X("Pred:N", axis=alt.Axis(title="Prédit", labelAngle=0)),
                                y=alt.Y("True:N", axis=alt.Axis(title="Vrai")),
                                color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges")),
                                tooltip=["True", "Pred", "Count"]
                            )
                            .properties(title="Matrice de confusion – binaire (test)", width=350, height=350)
                        )
                        st.altair_chart(chart_cm, use_container_width=True)
