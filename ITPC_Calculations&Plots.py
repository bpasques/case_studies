# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:49:37 2025

@author: User
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import mne
from mne.time_frequency import tfr_multitaper
import seaborn as sns

from mne.stats import permutation_cluster_test




#Make the ITPC average plots (averaged over channels and participants, so per session)
# Define frequencies of interest and time window
paths = [
    ("Session 1", r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\EpochedData\AllSubjects_Session1_TargetEpochs-epo.fif"),
    ("Session 2", r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\EpochedData\AllSubjects_Session2_TargetEpochs-epo.fif"),
    ("Session 3", r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\EpochedData\AllSubjects_Session3_TargetEpochs-epo.fif")
]

# Settings
channels = ['Pz', 'AFz', 'P4', 'P3', 'Oz', 'Cz']
freqs = np.linspace(1, 30, 30)      # 1–30 Hz
n_cycles = freqs / 2.               # Multitaper cycles
vmin = itpc_avg.min()
vmax = itpc_avg.max()
# Loop through sessions
for label, path in paths:
    print(f"Processing {label}")
    epochs = mne.read_epochs(path, preload=True)

    # Compute ITPC
    power, itpc = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=True,
        decim=2, average=True, picks=channels
    )

    # Average ITPC across selected channels
    itpc_avg = itpc.data.mean(axis=0)  # shape: (freqs, times)

    # Plot
    plt.figure(figsize=(8, 4))
    t = itpc.times * 1000  # ms
    f = itpc.freqs
    plt.imshow(itpc_avg, aspect='auto', origin='lower',
               extent=[t[0], t[-1], f[0], f[-1]],
               cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='ITPC')
    plt.title(f'ITPC Averaged over Channels ({", ".join(channels)})\n{label}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.axvline(0, color='white', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()
    
    

#Comparing the three sessions with cluster based approach
# Configuration
data_dir = r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\Processed"
subjects = [f"Subject{subj:02d}" for subj in range(1, 43)]
sessions = [1, 2, 3]
channels = ['Pz', 'Cz', 'AFz', 'P3', 'P4', 'Oz']
freqs = np.linspace(1, 30, 30)
n_cycles = freqs / 2.

# Prepare storage
itpc_data = {session: [] for session in sessions}
time_vector = None

# Extract ITPC per subject and session
for session in sessions:
    for subj in subjects:
        try:
            filename = f"{subj}_Session{session}_TargetEpochs-epo.fif"
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path):
                continue

            epochs = mne.read_epochs(path, preload=True)
            power, itpc = tfr_multitaper(
                epochs, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=True,
                average=True, picks=channels
            )

            if time_vector is None:
                time_vector = itpc.times * 1000  # convert to ms

            itpc_avg = itpc.data.mean(axis=0)  # average over channels (freqs x times)
            itpc_data[session].append(itpc_avg)

        except Exception as e:
            print(f"Skipping {subj} in session {session}: {e}")

# Convert lists to numpy arrays
for session in sessions:
    itpc_data[session] = np.array(itpc_data[session])  # shape: (n_subjects, n_freqs, n_times)

# Check the resulting shapes
{f"Session {session}": data.shape for session, data in itpc_data.items()}

{'Session 1': (0,), 'Session 2': (0,), 'Session 3': (0,)}

session_pairs = [(1, 2), (1, 3), (2, 3)]
cluster_results = {}

for s1, s2 in session_pairs:
    X1 = itpc_data[s1]
    X2 = itpc_data[s2]

    # Match lengths (skip subjects with missing data)
    min_n = min(len(X1), len(X2))
    X1 = X1[:min_n]
    X2 = X2[:min_n]

    X = [X1, X2]  # (n_subjects, n_freqs, n_times)
    print(f"Running cluster test: Session {s1} vs Session {s2}")
    T_obs, clusters, p_vals, _ = permutation_cluster_test(
        X, n_permutations=1000, tail=0, threshold=None, n_jobs=1, out_type='mask')

    sig_clusters = [(i, p) for i, p in enumerate(p_vals) if p < 0.05]

    cluster_results[(f"Session{s1}", f"Session{s2}")] = {
        "T_obs": T_obs,
        "clusters": clusters,
        "p_values": p_vals,
        "significant": sig_clusters
    }

session_pairs = [('Session1', 'Session2'), ('Session1', 'Session3'), ('Session2', 'Session3')]
print({k: len(v["significant"]) for k, v in cluster_results.items()})

for pair in session_pairs:
    T_obs = cluster_results[pair]['T_obs']
    clusters = cluster_results[pair]['clusters']
    p_values = cluster_results[pair]['p_values']

    # Create a significance mask
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i_c, p_val in enumerate(p_values):
        if p_val < 0.05:
            sig_mask[clusters[i_c]] = True

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(T_obs, aspect='auto', origin='lower',
               extent=[time_vector[0], time_vector[-1], freqs[0], freqs[-1]],
               cmap='RdBu_r')
    plt.colorbar(label='t-statistic')
    plt.contour(sig_mask, levels=[0.5], colors='black',
                linewidths=1.5, extent=[time_vector[0], time_vector[-1], freqs[0], freqs[-1]])

    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Cluster-Based Permutation Test: {pair[0]} vs {pair[1]}")
    plt.tight_layout()
    plt.show()





# Get data
X1 = itpc_data[1]
X3 = itpc_data[3]

# Get the significant cluster mask
sig_clusters = cluster_results[('Session1', 'Session3')]['significant']
cluster_index = sig_clusters[0][0]
cluster_mask = cluster_results[('Session1', 'Session3')]['clusters'][cluster_index]

# Masking and computing ITPC means and SDs
X1_cluster_values = X1[:, cluster_mask]
X3_cluster_values = X3[:, cluster_mask]

mean_s1 = X1_cluster_values.mean()
std_s1 = X1_cluster_values.std()

mean_s3 = X3_cluster_values.mean()
std_s3 = X3_cluster_values.std()

(mean_s1, std_s1, mean_s3, std_s3)
















