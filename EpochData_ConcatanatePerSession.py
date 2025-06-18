# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:39:05 2025

@author: User
"""

import os
import mne

# Path to directory with .fif files
data_dir = r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\Processed"
n_subjects = 42
session_numbers = [1, 2, 3]

# Store epochs for each session
all_epochs_session1 = []
all_epochs_session2 = []
all_epochs_session3 = []

# References for channel order
ref_info_sess1 = ref_info_sess2 = ref_info_sess3 = None

# Loop through all subjects and sessions
for subj in range(1, n_subjects + 1):
    for sess in session_numbers:
        filename = f"Subject{subj:02d}_Session{sess}_raw.fif"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            continue

        print(f"Loading {filename}")
        raw = mne.io.read_raw_fif(filepath, preload=True)

        # Find events and filter for target only (code 2)
        events_all = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
        events_target = events_all[events_all[:, 2] == 2]

        if len(events_target) == 0:
            print(f"⚠️ No target events in {filename}, skipping...")
            continue

        # Epoching (only EEG, with baseline correction)
        epochs = mne.Epochs(
            raw.copy().pick("eeg"),
            events=events_target,
            event_id={'Target': 2},
            tmin=-0.2,
            tmax=0.8,
            baseline=(None, 0),
            preload=True,
            reject_by_annotation=True
        )

        # Set channel order reference once per session group
        if sess == 1 and not all_epochs_session1:
            ref_info_sess1 = epochs.info
        elif sess == 2 and not all_epochs_session2:
            ref_info_sess2 = epochs.info
        elif sess == 3 and not all_epochs_session3:
            ref_info_sess3 = epochs.info

        # Reorder channels to match the reference
        if sess == 1:
            epochs = epochs.copy().reorder_channels(ref_info_sess1['ch_names'])
            all_epochs_session1.append(epochs)
        elif sess == 2:
            epochs = epochs.copy().reorder_channels(ref_info_sess2['ch_names'])
            all_epochs_session2.append(epochs)
        elif sess == 3:
            epochs = epochs.copy().reorder_channels(ref_info_sess3['ch_names'])
            all_epochs_session3.append(epochs)

# --- Concatenate epochs per session ---
print(" Concatenating session 1")
epochs_session1_all = mne.concatenate_epochs(all_epochs_session1)
print("Concatenating session 2")
epochs_session2_all = mne.concatenate_epochs(all_epochs_session2)
print(" Concatenating session 3")
epochs_session3_all = mne.concatenate_epochs(all_epochs_session3)

# --- Save combined epochs ---
output_dir = data_dir  # Save in same location

epochs_session1_all.save(os.path.join(output_dir, "AllSubjects_Session1_TargetEpochs-epo.fif"), overwrite=True)
epochs_session2_all.save(os.path.join(output_dir, "AllSubjects_Session2_TargetEpochs-epo.fif"), overwrite=True)
epochs_session3_all.save(os.path.join(output_dir, "AllSubjects_Session3_TargetEpochs-epo.fif"), overwrite=True)



print("All sessions processed, concatenated, and saved.")