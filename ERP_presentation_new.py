# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:37:22 2025

@author: bazil
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 19:53:19 2025
This is a script for advanced presentation of the ERP waves
=> it starts with the code for ERP plotting for ERP (but wuthout plotting the ITCP)

Goals for this script (inspired by the figures from the JNS Rousselet 16 paper)
=> add the confidence intervals and signficance points (B) 
=> for each condition comparison, add the difference wave plots (C&D)
=> for each condition comparison add a timecourse of the individual differences plot (F)
=> optionally: add a single participant difference wave plot for illustration (E)

CURRENT ISSUES: still the significant differences locations in the F panel figure do not seem to correspond with those calculated at subject level in E 

@author: bazil
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_rel
from scipy.stats import sem, t
import mne
from mne.time_frequency import tfr_multitaper
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test

# ------------------- CONFIGURATION -------------------
data_dir = r"C:\Users\bazil\OneDrive\Master\Case_studies\full_data\preprocessed"
n_subjects = 43  
sessions = [1, 2, 3]
sfreq = 512  # Sampling frequency, only used for time conversion
tmin, tmax = -0.2, 0.8  # Epoch window
event_id = {'Target': 2}
stim_channel = 'STI 014'

# ------------------- ERP COMPUTATION -------------------
session_evokeds = {sess: [] for sess in sessions}
session_epochs = {sess: [] for sess in sessions}  # <-- Initialize outside the loop

for subj in range(1, n_subjects + 1):
    subj_id = f"subject_{subj:02d}"
    subj_file_str = f"Subject{subj:02d}"

    for sess in sessions:
        filename = f"{subj_file_str}_Session{sess}_raw.fif"
        filepath = os.path.join(data_dir, subj_id, filename)

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue

        print(f"✅ Processing {subj_file_str} Session {sess}")

        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel=stim_channel, verbose=False)
        events_target = events[events[:, 2] == event_id['Target']]
        if len(events_target) == 0:
            print(f"⚠️ No target events for {subj_file_str} Session {sess}")
            continue

        epochs = mne.Epochs(
            raw.pick("eeg"),
            events_target,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )

        session_epochs[sess].append(epochs)  # ✅ Now storing trial-level data
        session_evokeds[sess].append(epochs.average())


# ------------------- FIG A PLOTTING GRAND AVERAGE -------------------
plt.figure(figsize=(10, 5))

for sess in sessions:
    evokeds = session_evokeds[sess]
    if not evokeds:
        print(f"⚠️ No data for Session {sess}")
        continue

    # Compute grand average across subjects
    grand_avg = mne.grand_average(evokeds)

    # Average over channels
    avg_data = grand_avg.data.mean(axis=0)  # shape: (n_times,)
    times = grand_avg.times * 1000  # convert to ms

    label = f"Session {sess}"
    plt.plot(times, avg_data * 1e6, label=label)  # µV

# Formatting
plt.axvline(0, color='k', linestyle='--', label='Stimulus Onset')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title("Grand Average ERP (All Channels Averaged)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------- FIG B: GA with confidence -------------------

#helper function to compute confidence intervals 
def compute_cousineau_ci(data, ci=0.95):
    """
    Compute within-subject confidence intervals (Cousineau-Morey corrected).
    Expects data shape (n_subjects, n_timepoints).
    Returns: mean ERP (n_timepoints), CI half-width (n_timepoints)
    """
    n_subjects = data.shape[0]

    # Normalize: remove each subject's mean, add grand mean
    subj_means = data.mean(axis=1, keepdims=True)
    grand_mean = data.mean()
    normalized = data - subj_means + grand_mean

    # Compute standard error and t-based CI
    sems = sem(normalized, axis=0)
    t_val = t.ppf((1 + ci) / 2., n_subjects - 1)
    ci_halfwidth = sems * t_val

    return data.mean(axis=0), ci_halfwidth

def choose_color(session1, session2): 
    # Define color mappings
    color_map = {
        1: ("forestgreen", "lightgreen"),
        2: ("navy", "skyblue"),
        3: ("darkred", "salmon")
    }

    line_color_1, fill_color_1 = color_map.get(session1, ("black", "gray"))
    line_color_2, fill_color_2 = color_map.get(session2, ("black", "gray"))

    return line_color_1, fill_color_1, line_color_2, fill_color_2
    

#a functino to plot the conditions against each other with CI and signficance
def plot_panel_B(session1, session2):
    evokeds1 = session_evokeds[session1]
    evokeds2 = session_evokeds[session2]

    if len(evokeds1) != len(evokeds2):
        raise ValueError("Sessions must have the same number of subjects")

    times = evokeds1[0].times * 1000  # ms

    # Stack subject data: shape = (n_subjects, n_times)
    data1 = np.stack([evo.data.mean(axis=0) for evo in evokeds1])
    data2 = np.stack([evo.data.mean(axis=0) for evo in evokeds2])

    # Compute grand average + 95% CI for both sessions
    mean1, ci1 = compute_cousineau_ci(data1)
    mean2, ci2 = compute_cousineau_ci(data2)

    # Paired t-test at each time point
    t_vals, p_vals = ttest_rel(data1, data2, axis=0)
        
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title(f"ERP + 95% CI + Significance (Session {session1} vs Session {session2})")

    # Helper funcion for assigning colors 
    line_color_1, fill_color_1, line_color_2, fill_color_2 = choose_color(session1, session2)
    # Plot Session 1
    plt.plot(times, mean1 * 1e6, label=f"Session {session1}", color=line_color_1)
    plt.fill_between(times, (mean1 - ci1) * 1e6, (mean1 + ci1) * 1e6,
                     color= fill_color_1, alpha=0.2)

    # Plot Session 2
    plt.plot(times, mean2 * 1e6, label=f"Session {session2}", color=line_color_2)
    plt.fill_between(times, (mean2 - ci2) * 1e6, (mean2 + ci2) * 1e6,
                     color=fill_color_2, alpha=0.2)

    # Significant time points (p < 0.05)
    sig_mask = p_vals < 0.05
    y_offset = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.02
    plt.plot(times[sig_mask], [y_offset] * sig_mask.sum(), 'k.', markersize=6)

    # Formatting
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------- FIG C: Difference wave + CI + individual traces -------------------
def plot_panel_C(session1, session2):
    evokeds1 = session_evokeds[session1]
    evokeds2 = session_evokeds[session2]

    if len(evokeds1) != len(evokeds2):
        raise ValueError("Sessions must have the same number of subjects")

    times = evokeds1[0].times * 1000  # ms

    # Stack subject data: shape = (n_subjects, n_times)
    data1 = np.stack([evo.data.mean(axis=0) for evo in evokeds1])
    data2 = np.stack([evo.data.mean(axis=0) for evo in evokeds2])

    # Compute subject-wise difference wave
    diff_data = data1 - data2

    # Mean and CI of difference
    mean_diff, ci_diff = compute_cousineau_ci(diff_data)

    # Paired t-test for significance (optional)
    t_vals, p_vals = ttest_rel(data1, data2, axis=0)
    sig_mask = p_vals < 0.05

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title(f"ERP Difference (Session {session1} – Session {session2})")

    # Plot individual differences
    for subj_diff in diff_data:
        plt.plot(times, subj_diff * 1e6, color='lightgray', linewidth=1)

    # Plot mean difference
    plt.plot(times, mean_diff * 1e6, color='black', linewidth=2.5, label='Mean difference')

    # Plot 95% CI around mean difference
    plt.fill_between(times, (mean_diff - ci_diff) * 1e6, (mean_diff + ci_diff) * 1e6,
                     color='black', alpha=0.2)

    # Significant time points
    y_offset = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.02
    plt.plot(times[sig_mask], [y_offset] * sig_mask.sum(), 'k.', markersize=6)

    # Formatting
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude difference (µV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------- FIG E: on subject levvel it works! ---------------------

def plot_panel_E_subject_diff_signif(subject_id: int, session1: int, session2: int, alpha=0.05):
    subj_idx = subject_id - 1
    epochs1 = session_epochs[session1][subj_idx]
    epochs2 = session_epochs[session2][subj_idx]

    # Match trial counts via truncation
    min_trials = min(len(epochs1), len(epochs2))
    epochs1 = epochs1[:min_trials]
    epochs2 = epochs2[:min_trials]

    # Extract and average across channels
    data1 = epochs1.get_data().mean(axis=1)  # shape (n_trials, n_times)
    data2 = epochs2.get_data().mean(axis=1)

    diff = data1 - data2
    diff_mean = diff.mean(axis=0)
    diff_sem = sem(diff, axis=0)
    times = epochs1.times * 1000  # ms

    # T-test at each time point
    t_vals, p_vals = ttest_rel(data1, data2, axis=0)
    significant = p_vals < alpha

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, diff_mean * 1e6, label=f'Session {session1} - {session2}', color='black')
    plt.fill_between(times, (diff_mean - diff_sem) * 1e6,
                     (diff_mean + diff_sem) * 1e6,
                     color='gray', alpha=0.3, label='±1 SEM')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    # Highlight significant time points
    plt.scatter(times[significant], 
                np.zeros_like(times[significant]), 
                color='red', s=10, label='p < {:.2f}'.format(alpha))

    plt.title(f'ERP Difference (Subject {subject_id}) with Significance')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    


# ------------------- FIG F: the final figure from the JSN paper  ---------------------
#DISCLAAIMER: check if it was computed by subject level data ?? check this =>i think it is not possible sinc ethere is not a loop for looping through the trial to get the signifcance 
from matplotlib.colors import TwoSlopeNorm  # Add this at the top of your script

def plot_panel_F_subjectwise_significance(session1, session2, alpha=0.05, tmin_ms=0, tmax_ms=600):
    """
    Panel F with subject-specific significance masks (limited time window).
    Each subject's difference wave is tested using a within-subject t-test across trials.
    Transparent = non-significant; Opaque = significant.
    """
    evokeds1 = session_evokeds[session1]
    evokeds2 = session_evokeds[session2]
    epochs1 = session_epochs[session1]
    epochs2 = session_epochs[session2]

    if len(evokeds1) != len(evokeds2):
        raise ValueError(f"❌ Mismatched subject counts: {len(evokeds1)} vs {len(evokeds2)}")

    n_subjects = len(evokeds1)
    n_times = len(evokeds1[0].times)
    full_times = evokeds1[0].times * 1000  # ms

    diff_data = np.zeros((n_subjects, n_times))
    sig_mask_2d = np.zeros((n_subjects, n_times), dtype=bool)

    for subj_idx in range(n_subjects):
        ep1 = epochs1[subj_idx]
        ep2 = epochs2[subj_idx]

        # Match trial counts
        min_trials = min(len(ep1), len(ep2))
        ep1 = ep1[:min_trials]
        ep2 = ep2[:min_trials]

        data1 = ep1.get_data().mean(axis=1)  # (n_trials, n_times)
        data2 = ep2.get_data().mean(axis=1)

        diff = data1 - data2
        diff_mean = diff.mean(axis=0)
        t_vals, p_vals = ttest_rel(data1, data2, axis=0)

        diff_data[subj_idx] = diff_mean
        sig_mask_2d[subj_idx] = p_vals < alpha

        print(f"Subject {subj_idx+1:02d}: {np.sum(p_vals < alpha)} significant timepoints")

    # Apply time mask
    time_mask = (full_times >= tmin_ms) & (full_times <= tmax_ms)
    times = full_times[time_mask]
    diff_data = diff_data[:, time_mask]
    sig_mask_2d = sig_mask_2d[:, time_mask]

    # Convert to µV for display
    diff_data_uV = diff_data * 1e6

    # Set symmetric normalization centered at 0 for the colormap
    abs_max = np.max(np.abs(diff_data_uV))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap('RdBu_r')

    # Plot non-significant (transparent)
    masked_nonsig = np.ma.masked_where(sig_mask_2d, diff_data_uV)
    ax.imshow(masked_nonsig, aspect='auto', cmap=cmap, norm=norm,
              extent=[times[0], times[-1], 0.5, n_subjects + 0.5],
              origin='lower', alpha=0.25)

    # Plot significant (opaque)
    masked_sig = np.ma.masked_where(~sig_mask_2d, diff_data_uV)
    im = ax.imshow(masked_sig, aspect='auto', cmap=cmap, norm=norm,
                   extent=[times[0], times[-1], 0.5, n_subjects + 0.5],
                   origin='lower', alpha=1.0)

    # Horizontal lines between participants
    for subj_idx in range(1, n_subjects):
        ax.axhline(subj_idx + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    # Formatting
    plt.colorbar(im, ax=ax, label='ERP Difference (µV)')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title(f'Panel F: Subject-wise ERP Difference + Significance\nSession {session1} vs {session2}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Subject')
    ax.set_yticks(np.arange(1, n_subjects + 1, step=2))
    ax.set_xlim(tmin_ms, tmax_ms)
    plt.tight_layout()
    plt.show()

    print("✅ Finished plotting Panel F (limited to {}–{} ms).\n".format(tmin_ms, tmax_ms))

# ------------------- FIG G: my own figure for comparing consensus significance   ---------------------

def plot_panel_G_proportion_sign(session1, session2, alpha=0.05, threshold=0.5): #i have outcommented the treshold for now 
    """
    Plot the grand average difference wave with a horizontal bar indicating 
    how many participants showed a significant difference at each timepoint.

    threshold: float between 0 and 1 (e.g., 0.5 for 50%) to optionally highlight consensus-level timepoints.
    """
    epochs1 = session_epochs[session1]
    epochs2 = session_epochs[session2]

    n_subjects = len(epochs1)
    n_times = len(epochs1[0].times)
    times = epochs1[0].times * 1000  # ms

    diff_data = np.zeros((n_subjects, n_times))
    sig_mask_2d = np.zeros((n_subjects, n_times), dtype=bool)

    for subj_idx in range(n_subjects):
        ep1 = epochs1[subj_idx]
        ep2 = epochs2[subj_idx]

        # Match trial counts
        min_trials = min(len(ep1), len(ep2))
        ep1 = ep1[:min_trials]
        ep2 = ep2[:min_trials]

        data1 = ep1.get_data().mean(axis=1)  # (n_trials, n_times)
        data2 = ep2.get_data().mean(axis=1)

        diff = data1 - data2  # (n_trials, n_times)
        diff_mean = diff.mean(axis=0)
        t_vals, p_vals = ttest_rel(data1, data2, axis=0)

        diff_data[subj_idx] = diff_mean
        sig_mask_2d[subj_idx] = p_vals < alpha

    # Grand average
    grand_mean_diff = diff_data.mean(axis=0)

    # Count number of subjects significant at each timepoint
    n_sig = sig_mask_2d.sum(axis=0)

    # Thresholded mask (e.g. >= 50% of subjects significant)
    consensus_threshold = int(n_subjects * threshold)
    consensus_mask = n_sig >= consensus_threshold

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                           gridspec_kw={'height_ratios': [4, 1]})

    # Top: grand average difference
    ax[0].plot(times, grand_mean_diff * 1e6, color='black', label='Grand avg diff')
    ax[0].axhline(0, color='gray', linestyle='--')
    ax[0].axvline(0, color='gray', linestyle='--')
    ax[0].set_ylabel("Amplitude (µV)")
    ax[0].set_title(f"Grand Average Difference (Session {session1} – {session2})")
    ax[0].legend()
    ax[0].grid(True)

    # Bottom: bar of subject counts
    ax[1].bar(times, n_sig, width=2, color='skyblue', align='center', label='Subjects with p < α')
    #ax[1].plot(times, [consensus_threshold] * len(times), 'r--', label=f'{int(threshold*100)}% threshold')
    ax[1].set_ylabel("n sig")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylim(0, n_subjects)
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

### PANEL F & G integrated ###
def plot_panel_F_with_subject_counts(session1, session2, alpha=0.05, tmin_ms=0, tmax_ms=600):
    """
    Combines Panel F (subject-wise ERP difference heatmap) with Panel G-style plot (n significant participants over time).
    Uses fill_between to ensure perfect alignment of time scales between panels.
    """
    evokeds1 = session_evokeds[session1]
    evokeds2 = session_evokeds[session2]
    epochs1 = session_epochs[session1]
    epochs2 = session_epochs[session2]

    if len(evokeds1) != len(evokeds2):
        raise ValueError(f"❌ Mismatched subject counts: {len(evokeds1)} vs {len(evokeds2)}")

    n_subjects = len(evokeds1)
    n_times = len(evokeds1[0].times)
    full_times = evokeds1[0].times * 1000  # ms

    diff_data = np.zeros((n_subjects, n_times))
    sig_mask_2d = np.zeros((n_subjects, n_times), dtype=bool)

    for subj_idx in range(n_subjects):
        ep1 = epochs1[subj_idx]
        ep2 = epochs2[subj_idx]

        min_trials = min(len(ep1), len(ep2))
        ep1 = ep1[:min_trials]
        ep2 = ep2[:min_trials]

        data1 = ep1.get_data().mean(axis=1)  # (n_trials, n_times)
        data2 = ep2.get_data().mean(axis=1)

        diff = data1 - data2
        diff_mean = diff.mean(axis=0)
        t_vals, p_vals = ttest_rel(data1, data2, axis=0)

        diff_data[subj_idx] = diff_mean
        sig_mask_2d[subj_idx] = p_vals < alpha

    # Time window masking
    time_mask = (full_times >= tmin_ms) & (full_times <= tmax_ms)
    times = full_times[time_mask]
    diff_data = diff_data[:, time_mask]
    sig_mask_2d = sig_mask_2d[:, time_mask]

    # Count how many subjects are significant at each timepoint
    n_sig = sig_mask_2d.sum(axis=0)

    # Create figure with two panels
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                            gridspec_kw={'height_ratios': [5, 1]})

    
    cmap = plt.get_cmap('RdBu_r')

    # --- TOP PANEL: heatmap per subject ---
    ax = axs[0]
    masked_nonsig = np.ma.masked_where(sig_mask_2d, diff_data) * 1e6
    ax.imshow(masked_nonsig, aspect='auto', cmap=cmap,
              extent=[times[0], times[-1], 0.5, n_subjects + 0.5],
              origin='lower', alpha=0.25)

    masked_sig = np.ma.masked_where(~sig_mask_2d, diff_data) * 1e6
    im = ax.imshow(masked_sig, aspect='auto', cmap=cmap,
                   extent=[times[0], times[-1], 0.5, n_subjects + 0.5],
                   origin='lower', alpha=1.0)

    for subj_idx in range(1, n_subjects):
        ax.axhline(subj_idx + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.axvline(0, color='black', linestyle='--')
    ax.set_ylabel('Subject')
    ax.set_title(f'Subject-wise ERP Difference + Significance\nSession {session1} vs {session2}')
    ax.set_yticks(np.arange(1, n_subjects + 1, step=2))
    ax.set_xlim(times[0], times[-1])
    plt.colorbar(im, ax=ax, label='ERP Difference (µV)')

    # --- BOTTOM PANEL: n significant participants over time ---
    ax2 = axs[1]
    ax2.fill_between(times, n_sig, step='mid', color='steelblue', alpha=0.8)
    ax2.set_ylim(0, n_subjects)
    ax2.set_ylabel('n sig')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title("Number of participants with p < α")
    ax2.grid(True)
    ax2.set_xlim(times[0], times[-1])

    plt.tight_layout()
    plt.show()

    print(" Finished combined Panel F + G.")


#%% ### PLOTTING ALL WE NEED ####

#for A & B maybe a loop is possible 
plot_panel_B(1,2)
plot_panel_B(2,3)
plot_panel_B(1,3)

#plotting the C plots 
plot_panel_C(1, 2)
plot_panel_C(2, 3)
plot_panel_C(1, 3)

# here add some extra examples to support figure F; add a loop so that it happens for one participant across sessions 
participants = [6,24]
for p in participants: 
    plot_panel_E_subject_diff_signif(subject_id=p, session1=1, session2=2, alpha=0.05) #FIX that it loops over the different session combinations for participant 6 & 24
    plot_panel_E_subject_diff_signif(subject_id=p, session1=2, session2=3, alpha=0.05)
    plot_panel_E_subject_diff_signif(subject_id=p, session1=1, session2=3, alpha=0.05)
    
# inspect the individual differences in the time window of interest for diffrent conditions 
plot_panel_F_subjectwise_significance(1,2, tmin_ms=200, tmax_ms=600)
plot_panel_F_subjectwise_significance(2,3, tmin_ms=200, tmax_ms=600)
plot_panel_F_subjectwise_significance(1,3, tmin_ms=200, tmax_ms=600)
# => this allos us to see which of the individual participants where signifiant toogether (consesnus significance) and also how many of the participants went along with vs. against the main fx 

# Panel G to support panel F! => so place each one at it's respective F figure 
plot_panel_G_proportion_sign(1, 2, alpha=0.05, threshold=0.5) 
plot_panel_G_proportion_sign(2, 3, alpha=0.05, threshold=0.5) 
plot_panel_G_proportion_sign(1, 3, alpha=0.05, threshold=0.5) 


# acrually the trehsold i have left it out for now because i found no support for this in the literature 

# INTEGRATED F & G (might be enough ) => here is some timescale issue (skip for now, i cant seem to figure it out )
#plot_panel_F_with_subject_counts(1, 2, alpha=0.05, tmin_ms=200, tmax_ms=600)


