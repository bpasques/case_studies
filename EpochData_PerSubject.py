# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:19:18 2025

@author: User
"""

import os
import mne

# Path to directory with .fif files
data_dir = r"C:\Users\User\OneDrive - UGent\Bureaublad\Case Studies\Data\Processed"
n_subjects = 42
session_numbers = [1, 2, 3]

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

        # Set channel order reference once per session
        if sess == 1 and ref_info_sess1 is None:
            ref_info_sess1 = epochs.info
        elif sess == 2 and ref_info_sess2 is None:
            ref_info_sess2 = epochs.info
        elif sess == 3 and ref_info_sess3 is None:
            ref_info_sess3 = epochs.info

        # Reorder channels to match the reference
        if sess == 1:
            epochs = epochs.copy().reorder_channels(ref_info_sess1['ch_names'])
        elif sess == 2:
            epochs = epochs.copy().reorder_channels(ref_info_sess2['ch_names'])
        elif sess == 3:
            epochs = epochs.copy().reorder_channels(ref_info_sess3['ch_names'])

        # Save individual epoch file
        output_filename = f"Subject{subj:02d}_Session{sess}_TargetEpochs-epo.fif"
        output_path = os.path.join(data_dir, output_filename)
        epochs.save(output_path, overwrite=True)
        print(f"✅ Saved: {output_filename}")
