import argparse, re, sys
from typing import List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- arg parsing -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input", required=True, help="CSV input path")
    p.add_argument("-o","--output", default="final_styled_plot.html", help="HTML output path")
    p.add_argument("--show-count", type=int, default=8,
                   help="How many EEG channels to show by default (others hidden in legend).")
    return p.parse_args()

# ----------------- csv loading & detection -----------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, comment='#', header=0)

def find_time_col(cols: List[str]) -> str:
    for c in cols:
        if re.search(r'\btime\b', c, re.I): return c
    return cols[0]

def detect_channels(cols: List[str]):
    known_eeg = {"Fz","Cz","P3","C3","F3","F4","C4","P4","Fp1","Fp2",
                 "T3","T4","T5","T6","O1","O2","F7","F8","A1","A2","Pz"}
    ecg = [c for c in cols if re.search(r'(^X1\b|^X2\b|LEOG|REOG)', c, re.I)]
    cm = [c for c in cols if c.strip().upper() == "CM"]
    eegs = [c for c in cols if c in known_eeg]
    return {"ecg": ecg, "cm": (cm[0] if cm else None), "eeg_detected": eegs}

def fallback_eeg_columns(df: pd.DataFrame, time_col: str, ecg: List[str], cm: str):
    skip = set([time_col]) | set(ecg) | ({cm} if cm else set())
    skip |= {"Trigger","Time_Offset","ADC_Status","ADC_Sequence","Event","Comments","X3:"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

# ----------------- utilities -----------------
def compute_p2p_median(df: pd.DataFrame, channels: List[str]) -> float:
    p2p = []
    for ch in channels:
        arr = np.nan_to_num(df[ch].values)
        p2p.append(np.percentile(arr, 90) - np.percentile(arr, 10))
    p2p = np.array(p2p)
    med = float(np.median(p2p[p2p>0])) if np.any(p2p>0) else 1.0
    return med

def build_offsets_arrays(df: pd.DataFrame, channels: List[str], base_spacing: float):
    N = len(channels)
    offsets = [(N-1 - i) * base_spacing for i in range(N)]
    ys = [df[ch].values + offsets[i] for i,ch in enumerate(channels)]
    return ys, offsets

def build_normalized_arrays(df: pd.DataFrame, channels: List[str], norm_spacing: float):
    N = len(channels)
    offsets = [(N-1 - i) * norm_spacing for i in range(N)]
    ys = []
    for i,ch in enumerate(channels):
        arr = np.nan_to_num(df[ch].values)
        if np.std(arr) == 0:
            z = arr
        else:
            z = (arr - np.mean(arr)) / (np.std(arr) + 1e-12)
        ys.append(z + offsets[i])
    return ys, offsets

# ----------------- build figure -----------------
def build_figure(df: pd.DataFrame, time_col: str, eeg_chs: List[str], ecg_chs: List[str],
                 cm_col: str, show_count: int = 8):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.3, 0.7], vertical_spacing=0.06)

    # Neon ECG palette
    ecg_palette = ["#FF073A","#39FF14","#0FF0FC","#F5F500","#FF6EC7"]
    cm_color = "#00FFFF"  # CM distinct neon
    eeg_palette = [
        "#1abc9c","#2ecc71","#3498db","#9b59b6","#f39c12","#e74c3c","#e67e22",
        "#16a085","#27ae60","#2980b9","#8e44ad","#f1c40f","#c0392b","#d35400",
        "#7f8c8d","#bdc3c7","#34495e","#6c5ce7","#f78fb3","#00cec9"
    ]

    # ---------- Top: ECG + CM ----------
    for idx,ch in enumerate(ecg_chs):
        color = ecg_palette[idx % len(ecg_palette)]
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[ch],
            name=f"{ch} (ECG)", legendgroup=f"ECG_{ch}",
            mode='lines', line=dict(color=color, width=2.5),
            hovertemplate="%{x:.2f}s<br>%{y:.2f}"
        ), row=1, col=1)

    if cm_col:
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[cm_col],
            name=f"{cm_col} (CM)", legendgroup=f"CM_{cm_col}",
            mode='lines', line=dict(color=cm_color, dash='dash', width=2.8),
            hovertemplate="%{x:.2f}s<br>%{y:.2f}"
        ), row=1, col=1)

    fig.update_yaxes(title_text="ECG / CM (raw units)", row=1, col=1)

    # ---------- Bottom: EEG ----------
    base_med = compute_p2p_median(df, eeg_chs)
    spacing_multipliers = [0.8, 1.0, 1.6, 2.4, 3.6]
    base_spacing_values = [max(1.0, base_med * m) for m in spacing_multipliers]

    raw_all_levels = []
    for s in base_spacing_values:
        ys, offsets = build_offsets_arrays(df, eeg_chs, s)
        raw_all_levels.append(ys)

    norm_spacing = 3.0
    norm_ys, norm_offsets = build_normalized_arrays(df, eeg_chs, norm_spacing)
    initial_raw = raw_all_levels[1]

    for idx,ch in enumerate(eeg_chs):
        color = eeg_palette[idx % len(eeg_palette)]
        fig.add_trace(go.Scatter(
            x=df[time_col], y=initial_raw[idx],
            name=f"{ch} (EEG µV)", legendgroup=f"EEG_{ch}",
            visible=True, mode='lines',
            line=dict(color=color, width=1.9),
            hovertemplate="%{x:.2f}s<br>%{y:.2f}"
        ), row=2, col=1)

    fig.update_yaxes(title_text="EEG (stacked, µV)", row=2, col=1)

    # ---------- Controls ----------
    n_ecg = len(ecg_chs) + (1 if cm_col else 0)
    n_eeg = len(eeg_chs)
    total_traces = n_ecg + n_eeg
    eeg_trace_indices = list(range(n_ecg, n_ecg + n_eeg))

    slider_steps = []
    for lvl_index, ys_level in enumerate(raw_all_levels):
        step = dict(method="restyle",
                    args=[{"y": ys_level}, eeg_trace_indices],
                    label=f"{spacing_multipliers[lvl_index]:.2f}x")
        slider_steps.append(step)

    raw_button = dict(label="Raw (µV)", method="restyle",
                      args=[{"y": initial_raw}, eeg_trace_indices])
    norm_button = dict(label="Normalized (z-score)", method="restyle",
                       args=[{"y": norm_ys}, eeg_trace_indices])

    vis_show_all = [True]*total_traces
    vis_show_eeg = [False]*n_ecg + [True]*n_eeg
    vis_show_ecg = [True]*n_ecg + [False]*n_eeg
    vis_hide_all = [False]*total_traces

    ui_bg = "rgba(245,247,250,0.95)"      
    button_bg = "rgb(205, 228, 255)"   
    button_border = "rgba(200,210,220,0.9)"
    slider_bg = "rgba(245,247,250,0.9)"
    legend_bg = "rgba(255,255,255,0.85)"
    paper_bg = "white"
    plot_bg = "rgba(250,252,254,1)"

    updatemenus = [
        dict(type="buttons", direction="right", x=0.02, y=1.16, xanchor="left", showactive=True,
             bgcolor=button_bg, bordercolor=button_border, pad=dict(r=6,t=6), font=dict(size=12),
             buttons=[raw_button, norm_button]),
        dict(type="buttons", direction="right", x=0.40, y=1.16, xanchor="left", showactive=True,
             bgcolor=button_bg, bordercolor=button_border, pad=dict(r=6,t=6), font=dict(size=12),
             buttons=[
                 dict(label="Show All", method="update", args=[{"visible": vis_show_all}, {"title":"Show All"}]),
                 dict(label="Show EEG", method="update", args=[{"visible": vis_show_eeg}, {"title":"Show EEG"}]),
                 dict(label="Show ECG", method="update", args=[{"visible": vis_show_ecg}, {"title":"Show ECG"}]),
                 dict(label="Hide All", method="update", args=[{"visible": vis_hide_all}, {"title":"Hide All"}])
             ])
    ]

    slider = dict(active=1,
                  currentvalue={"prefix":"EEG spacing: "},
                  pad={"t": 50, "b": 10},
                  steps=slider_steps,
                  x=0.02, y=0.02, len=0.96)

    fig.update_layout(
        title=dict(
            text="Interactive Multichannel EEG + ECG Time-Series Plot",
            x=0.5,     
            y=0.98,     
            xanchor='center',
            yanchor='top',
            font=dict(family="Arial, Helvetica, sans-serif", size=16, color="#111111")
        ),
        font=dict(family="Segoe UI, Arial, Helvetica, sans-serif", size=13, color="#152238"),
        margin=dict(l=120, r=180, t=120, b=90),
        height=780,
        updatemenus=updatemenus,
        sliders=[slider],
        dragmode="zoom",
        legend=dict(
            title="Channels",
            font=dict(family="Segoe UI, Arial, Helvetica, sans-serif", size=11, color="#152238"),
            bgcolor="rgb(205, 228, 255)",
            bordercolor="rgba(200,210,220,0.9)",
            borderwidth=0.5,
            traceorder="normal",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
        ),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers"
    )

    fig.add_annotation(dict(
        text="Buttons: Raw/Normalized & group toggles. Slider: EEG spacing. Legend: click to hide/show channels. Use toolbar to zoom/reset.",
        xref="paper", yref="paper", x=0.02, y=1.09, showarrow=False, align="left",
        font=dict(size=11, color="#333333"), bgcolor=ui_bg, bordercolor="rgba(0,0,0,0.03)",
        borderpad=6
    ))

    return fig

# ----------------- main -----------------
def main():
    args = parse_args()
    try:
        df = load_csv(args.input)
    except Exception as e:
        print("Failed to load CSV:", e, file=sys.stderr)
        sys.exit(1)

    cols = list(df.columns)
    time_col = find_time_col(cols)
    ch = detect_channels(cols)
    ecg_chs = ch["ecg"]
    cm_col = ch["cm"]
    eeg_detected = ch["eeg_detected"]
    if not eeg_detected:
        eeg_chs = fallback_eeg_columns(df, time_col, ecg_chs, cm_col)
    else:
        eeg_chs = eeg_detected

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if len(eeg_chs) == 0 and len(ecg_chs) == 0:
        print("No EEG or ECG channels found. Exiting.", file=sys.stderr)
        sys.exit(1)

    fig = build_figure(df, time_col, eeg_chs, ecg_chs, cm_col, show_count=args.show_count)
    fig.write_html(args.output, include_plotlyjs='cdn')
    print("Wrote", args.output)

if __name__ == "__main__":
    main()