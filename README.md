# QUASAR Coding Screener — Interactive Multichannel EEG + ECG Plot

## Overview
This Python project loads multichannel EEG and ECG data from a CSV file and generates an interactive, scrollable, and zoomable Plotly visualization. 

## Features
- Loads CSV data while ignoring metadata lines starting with `#`.
- Automatically detects EEG channels, ECG channels, and a reference/CM channel (plotted as a dashed line).
- Two-panel plot:
  - **Top panel:** ECG + CM
  - **Bottom panel:** Stacked EEG channels
- Raw vs Normalized (z-score) toggle for EEG
- EEG spacing slider
- Group visibility buttons: Show All / Show EEG / Show ECG / Hide All
- Zooming with Box Zoom, plus traditional zoom in/out
- Panning, autoscale, and reset axes
- Selectively show/hide channels
- Download a PNG snapshot of the plot

## Dependencies
- Python 3.8+
- pandas
- numpy
- plotly

Install dependencies using:
```bash
pip install pandas numpy plotly
```

## Usage
python plot_eeg_ecg.py -i "EEG and ECG data_02_raw.csv" -o plot.html

## Design Notes
- EEG signals (µV) are stacked and optionally normalized for better comparison.
- ECG signals (mV) are plotted separately but aligned.
- CM is plotted as a dashed line for reference only.
- UI design emphasizes readability and usability with interactive controls and color coding.

## Future Work
- Integrate unit conversion options (µV ↔ mV) for flexible comparison.
- Improve the design visually by finding a way to add background colors to each element to make them more distinguishable and more aesthetically appealing.
- Add a feature to let user choose colors of each line.
- Explore better techniques to provide more visual separation between EEG traces, enhancing clarity when multiple channels are displayed.

