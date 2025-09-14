# SankeyX with Dynamic States

Interactive Streamlit app to visualize clickstream sequences as a **Sankey diagram** with **dynamic customer intent timelines**, **SHAP‑weighted flows**, outcome → **utility**, and optional **AI explanations** via a local Ollama LLM.

![Architecture](toolarch.png)


![Dynamic States Example](session_timeline_hysteresis.gif)

---

## ✨ Features
- **CSV → Interactive Sankey**: maps each session's steps into a left‑to‑right flow with stage headers (Intent → Click 1..k → Prediction → Utility).
- **Dynamic Intent Engine**: rule‑based, prefix‑evaluated intents with optional hysteresis (K) and NA bridging to stabilize state switches.
- **Per‑Session Mode**: click a specific flow to render a **GIF timeline** of intents (matplotlib animation).
- **SHAP‑Weighted Links**: positive/negative step contributions change link width (and color in the UI).
- **LLM Insights (optional)**: send a compact summary to **Ollama** to get JSON insights, strengths/weaknesses, patterns, and tips.
- **Rich Sidebar Controls**: session count, max steps, ordering (first/last), event colors, intent strategy (late vs. hysteresis), etc.

> This README is generated based on the code you provided. Paths, columns, and behaviors reflect the current implementation.

---

## 📦 Input Data Schema (CSV)
Minimal columns:
- `truncated_sequence` — list‑like string of ints (e.g., `"[1,1,2,3,5]"`).
- `SHAP_1 ... SHAP_N` — per‑step SHAP values (N should cover the longest shown sequence).
- `y_pred` — model predicted label (0/1).
- `purchase` — ground truth label (0/1).
- `Intent_type` *(optional)* — static/classic intent label used for filtering.

> The app will **parse** `truncated_sequence`, **align** SHAP to visible steps, and **truncate at the first purchase (5)** if present.

---

# SankeyX — E-commerce

**Storyline:** We predict **purchase (1/0)** from clickstreams *(browse → detail → add → abandon → purchase)*. **SankeyX** reveals **buyer paths** like **detail → add → purchase** driving **TP**, while **browse↔detail loops → abandon** explain **TN/FN**, and **add ↔ abandon ping-pong** without checkout appears in **FP**. The **dynamic intent timeline** shows shifts—*Hesitant → Exploratory → Comparative → Decisive* for buyers, versus *Hesitant/Exploratory → Drop* for non-buyers. With a **utility matrix** balancing **conversion profit** vs **incentive/remarketing cost**, the tool highlights **where to nudge** (e.g., repeated detail + add within-session) and **where to suppress offers** (long wandering with no add), making **targeting and ROI** decisions explicit.



# SankeyX — Manufacturing

**Storyline:** We predict **defect / rework risk (1/0)** from process event sequences *(setup → run → inspection → rework → good_output)*. With **SankeyX**, we see which **subsequences drive alarms**: tight **run → inspection → rework** loops dominate **TP** (true issues), long **run** stretches with **late inspection** explain **FN** (missed defects), and **inspection-only repeats** without rework account for **FP** (over-triggering). The **dynamic process-health timeline** surfaces shifts—*Stable → Drift Suspected → At-Risk → Rework Loop*—and a **tunable utility matrix** (heavy penalty on FN: scrap/field failures; moderate on FP: unnecessary stops) **prioritizes lots/lines** where one intervention yields the biggest **scrap reduction** (e.g., earlier gates, tighter rework criteria).


# SankeyX — Clinical Pathway

**Storyline:** We predict **treatment escalation / progression (1/0)** from patient event sequences *(consult → test → treatment → follow-up → recovery)*. Using **SankeyX**, we see which **subsequences drive decisions**: patterns like **treatment → test soon after treatment** and **frequent re-staging** dominate **TP**, while **stacked follow-ups without returning to treatment** often cause **FN**; **toxicity-driven interruptions** explain many **FP**. The **dynamic clinical-state timeline** makes shifts visible—*Stable/Rising → Progression Suspected* before TP, *Toxicity Interruptions* before FP, and *Remission* when recovery appears. With a **tunable utility matrix** that heavily penalizes FN, the tool **prioritizes who to act on first**, highlighting cohorts where preventing one miss yields the biggest gain.





## 🚀 Quickstart

### 1) Install
```bash
pip install streamlit plotly pandas numpy matplotlib requests
# (optional) for click capture:
pip install streamlit-plotly-events
```

### 2) Run
```bash
streamlit run SankeyX_with_intent_dynamics.py
```

### 3) Upload Data
- Use the **sidebar** to upload your CSV.
- Tune: **N Sessions**, **Max Steps**, **Order (first/last)**, **event colors**, and **Intent Dynamics**.

### 4) Optional: Enable Ollama (local LLM)
- Install Ollama and pull a model:
  ```bash
  ollama run mistral      # or llama3.2, qwen2.5, etc.
  ```
- In the sidebar, set **Model** and toggle **Auto‑generate explanation** or click the button.

---

## ⚙️ Intent Dynamics (Rule Summary)
- Evaluate **prefixes** of the sequence at each step.
- Choose the currently “true” intent by stage ordering (**late**) or require **K consecutive confirmations** (**hysteresis**).
- **Bridge NA**: when no rule fires, optionally carry forward the last valid intent to avoid gaps.
- The **left column** can show either the **last dynamic intent** (default) or the **static intent** from the CSV.

---

## 🖼️ Outputs
- **SankeyX** (Plotly): per‑session flows, stage headers, outcomes (**TP/TN/FP/FN**), and final **Utility**.
- **On Click** (per‑session mode): a **GIF** of the dynamic intent timeline is generated and shown.
- **AI Explanation**: a tabbed JSON‑to‑paragraph view with performance, patterns, SHAP interpretation, and actionable insights.

> GIFs are saved to your user directory with a unique filename.

---

## 🔧 Troubleshooting
- **“Please upload a clickstream CSV file to begin.”** → No file was provided.
- **Missing column** → Ensure `truncated_sequence` and SHAP columns exist; `Intent_type` is optional.
- **Ollama error** → Start a local server: `ollama run mistral`. Check `http://localhost:11434` is reachable.
- **No GIF on click** → Use *Separate Sessions* mode and click a **flow/link** (not only nodes).

---

## 🧪 Tips
- Keep `Max Steps` small (e.g., 5–11) for a compact, readable Sankey.
- The app **truncates at the first purchase** (5) by design to focus on pre‑purchase logic.
- Adjust **SHAP ×** multiplier in the sidebar to accentuate link widths.

---

## 🪪 License
MIT.

## 🙌 Contributing
PRs welcome! Please open an issue for bugs/ideas and include a small CSV snippet to reproduce.

---

**Files included in this package**
- `SankeyX_with_intent_dynamics.py` — Streamlit app
- `README.md` — this file
- `sankeyx_architecture.png` — high‑level architecture diagram
