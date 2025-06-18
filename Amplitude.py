# -*- coding: utf-8 -*-
"""
Optimized ERP and ITPC analysis script
June 2025 => this is now the more modularized version of thescript so that i can run sections separately 
@author: bazil
UPDATE: it works equivalently to the old script! so we can continue with this one 
DISCLAIMER: in case that we want ITCP from this script we have to fix some code 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import mne
from mne.time_frequency import tfr_multitaper
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# --- Configuration ---
data_dir = r"C:\Users\bazil\OneDrive\Master\Case_studies\full_data\preprocessed"
n_subjects = 43
sessions = [1, 2, 3]
tmin, tmax = -0.2, 0.8
freqs = np.linspace(1, 30, 30)
n_cycles = freqs / 2.
itpc_window = (0.0, 0.6)

# --- Stage 1: Load and store all evoked data ---
print("\n=== Stage 1: Loading and preprocessing data ===")
evoked_data = {}  # key: (subject, session), value: evoked object

for subj in range(1, n_subjects + 1):
    subj_id_str = f"subject_{subj:02d}"
    subj_file_str = f"Subject{subj:02d}"
    for sess in sessions:
        filename = f"{subj_file_str}_Session{sess}_raw.fif"
        filepath = os.path.join(data_dir, subj_id_str, filename)

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue

        print(f" Loading {subj_file_str} Session {sess}")
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        events_all = mne.find_events(raw, stim_channel='STI 014', shortest_event=1, verbose=False)
        events_target = events_all[events_all[:, 2] == 2]
        if len(events_target) == 0:
            print(f"No target events found")
            continue

        epochs = mne.Epochs(
            raw.copy().pick("eeg"),
            events=events_target,
            event_id={'Target': 2},
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )

        evoked = epochs.average()
        evoked_data[(subj, sess)] = evoked

print("\n Stage 1 complete: All evoked data loaded.\n")

#for debugging
print(f" Subjects included: {sorted(set([key[0] for key in evoked_data.keys()]))}")
print(f" Sessions included: {sorted(set([key[1] for key in evoked_data.keys()]))}")


# === Stage 2: Analysis (adjust parameters freely here) ===
print("=== Stage 2: Running flexible analysis ===")

# Parameters you can adjust without reloading
amplitude_window = (0.3, 0.6)  # P300 window
print(f" Using amplitude window: {amplitude_window[0]}–{amplitude_window[1]} seconds")

compute_itpc = False  # Set to False to skip ITPC (for speed)

# --- Initialize results ---
erp_scores = []
itpc_scores = []

for (subj, sess), evoked in evoked_data.items():
    avg_data = evoked.data.mean(axis=0)
    times = evoked.times
    win_mask = (times >= amplitude_window[0]) & (times <= amplitude_window[1])

    # Mean and peak amplitude
    mean_amp = avg_data[win_mask].mean() * 1e6  # µV
    peak_amp = avg_data[win_mask].max() * 1e6  # µV 
    
    # Latency calculation 
    peak_idx = np.argmax(avg_data[win_mask])
    peak_latency = times[win_mask][peak_idx] * 1000  # Convert to ms

    #calculation of the peak amplitude will both serve for the peak amplitude and latency analyses 

    erp_scores.append({
        'subject': subj,
        'session': sess,
        'erp_mean_amplitude': mean_amp,
        'erp_peak_amplitude': peak_amp, 
        'erp_peak_latency': peak_latency  # in milliseconds
    })

    # ITPC (optional)
    if compute_itpc:
        epochs = evoked.to_epochs()
        _, itpc = tfr_multitaper(
            epochs, freqs=freqs, n_cycles=n_cycles,
            use_fft=True, return_itc=True,
            average=True, picks="eeg", verbose=False
        )
        t_idx = np.where((itpc.times >= itpc_window[0]) & (itpc.times <= itpc_window[1]))[0]
        itpc_mean = itpc.data[:, :, t_idx].mean()
        itpc_scores.append({'subject': subj, 'session': sess, 'itpc_score': itpc_mean})

#compare ERP values of a few subjects for DEBUGGING
for entry in erp_scores:
    if entry['subject'] in [1, 2]:  # choose key subjects
        print(f"Subject {entry['subject']} Session {entry['session']}: Mean Amp = {entry['erp_mean_amplitude']:.3f} µV, Peak Amp = {entry['erp_peak_amplitude']:.3f} µV")


# --- Convert to DataFrames ---
df_erp = pd.DataFrame(erp_scores)
pivot_erp_mean = df_erp.pivot(index='subject', columns='session', values='erp_mean_amplitude')
pivot_erp_mean.columns = ['Session1', 'Session2', 'Session3']

pivot_erp_peak = df_erp.pivot(index='subject', columns='session', values='erp_peak_amplitude')
pivot_erp_peak.columns = ['Session1', 'Session2', 'Session3']

pivot_erp_latency = df_erp.pivot(index='subject', columns='session', values='erp_peak_latency')
pivot_erp_latency.columns = ['Session1', 'Session2', 'Session3']


if compute_itpc:
    df_itpc = pd.DataFrame(itpc_scores)
    pivot_itpc = df_itpc.pivot(index='subject', columns='session', values='itpc_score')
    pivot_itpc.columns = ['Session1', 'Session2', 'Session3']

# --- Paired t-tests ---
def paired_t_tests(pivot_df, label):
    t12, p12 = ttest_rel(pivot_df['Session1'], pivot_df['Session2'], nan_policy='omit')
    t13, p13 = ttest_rel(pivot_df['Session1'], pivot_df['Session3'], nan_policy='omit')
    t23, p23 = ttest_rel(pivot_df['Session2'], pivot_df['Session3'], nan_policy='omit')
    df_stats = pd.DataFrame({
        'Comparison': ['Session1 vs Session2', 'Session1 vs Session3', 'Session2 vs Session3'],
        't-value': [t12, t13, t23],
        'p-value': [p12, p13, p23]
    })
    print(f"\n=== Paired t-test Results: {label} ===")
    print(df_stats)
    return df_stats

df_amp_stats = paired_t_tests(pivot_erp_mean, "ERP Mean Amplitude")
df_peak_stats = paired_t_tests(pivot_erp_peak, "ERP Peak Amplitude")
df_latency_stats = paired_t_tests(pivot_erp_latency, "ERP Peak Latency")

if compute_itpc:
    df_itpc_stats = paired_t_tests(pivot_itpc, "ITPC")

# --- Repeated Measures ANOVA ---
def run_anova(df, value_column, label):
    df_anova = df.pivot(index='subject', columns='session', values=value_column).reset_index()
    df_long = pd.melt(df_anova, id_vars=['subject'], value_vars=[1, 2, 3],
                      var_name='session', value_name='value')
    #debugging
    print(f" Melting over sessions: {[1, 2, 3]}")
    
    df_long['session'] = df_long['session'].astype(str)
    aovrm = AnovaRM(df_long, depvar='value', subject='subject', within=['session'])
    res = aovrm.fit()
    print(f"\n--- Repeated Measures ANOVA: {label} ---")
    print(res.summary())


#run actual anova 
run_anova(df_erp, 'erp_mean_amplitude', "ERP Mean Amplitude")
run_anova(df_erp, 'erp_peak_amplitude', "ERP Peak Amplitude")
run_anova(df_erp, 'erp_peak_latency', "ERP Peak Latency")

# --- Plots ---
# mean erp 
plt.figure(figsize=(7, 5))
sns.boxplot(data=pivot_erp_mean, palette='Set3')
sns.swarmplot(data=pivot_erp_mean, color='black', size=4)
plt.ylabel('ERP Mean Amplitude (µV, 300–600 ms)')
plt.title('ERP Mean Amplitudes per Session')
plt.tight_layout()
plt.grid(True)
plt.show()

# peak erp 
plt.figure(figsize=(7, 5))
sns.boxplot(data=pivot_erp_peak, palette='Set1')
sns.swarmplot(data=pivot_erp_peak, color='black', size=4)
plt.ylabel('ERP Peak Amplitude (µV, 300–600 ms)')
plt.title('ERP Peak Amplitudes per Session')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=pivot_erp_latency, palette='coolwarm')
sns.swarmplot(data=pivot_erp_latency, color='black', size=4)
plt.ylabel('ERP Peak Latency (ms, 300–600 ms window)')
plt.title('ERP Peak Latency per Session')
plt.tight_layout()
plt.grid(True)
plt.show()

if compute_itpc:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=pivot_itpc, palette='Set2')
    sns.swarmplot(data=pivot_itpc, color='black', size=4)
    plt.ylabel('ITPC Score (0–600 ms, all channels)')
    plt.title('ITPC Scores per Session')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

print("\n Finished: All analyses completed.")

#### PART 2 some addditions for the pairwise figures ####

# Plot for mean & peak amplitude and for peak latency 
dependent_vars = ['erp_mean_amplitude', 'erp_peak_amplitude', 'erp_peak_latency']

# create forloop for plotting 
for VAR in dependent_vars:
    df_long = df_erp[['subject', 'session', VAR]].copy()
    df_long['session'] = df_long['session'].astype(str)

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_long, x='session', y=VAR, hue='subject',
                 marker='o', palette='tab20', linewidth=1, alpha=0.5, legend=False)
    plt.title(f'Within-Subject {VAR}')
    plt.xlabel('Session')
    
    # Label y-axis based on the variable
    if 'latency' in VAR:
        plt.ylabel('Peak Latency (ms)')
    else:
        plt.ylabel('Amplitude (µV)')
        
    plt.grid(True)
    plt.tight_layout()
    plt.show()

