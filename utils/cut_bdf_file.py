import mne

raw = mne.io.read_raw_bdf('data/raw/V00_data/V06_data.bdf', preload=True)

raw.crop(tmin=10, tmax=20)

mne.export.export_raw('V06_data.bdf', raw, fmt='bdf', overwrite=True)