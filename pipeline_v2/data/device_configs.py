"""
Consumer device channel layouts for simulated sparse-headset experiments.

All channel names follow MNE standard_1005 convention — these are the names
that will appear in the .fif files after loading PhysioNet and renaming to
match the montage.  They are case-sensitive.

The 64 channels below are exactly what PhysioNet EEGMMIDB contains, expressed
in standard_1005 names (e.g. PhysioNet's 'Af3.' → standard_1005 'AF3').
"""

from typing import Dict, List

# ─── Full 64-channel list (PhysioNet EEGMMIDB → standard_1005 names) ─────────
PHYSIONET_64_CHANNELS: List[str] = [
    # Fronto-central
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    # Central
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    # Centro-parietal
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    # Frontal-polar
    "Fp1", "Fpz", "Fp2",
    # Anterior-frontal
    "AF7", "AF3", "AFz", "AF4", "AF8",
    # Frontal
    "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
    # Fronto-temporal
    "FT7", "FT8",
    # Temporal
    "T7",  "T8",  "T9",  "T10",
    # Temporo-parietal
    "TP7", "TP8",
    # Parietal
    "P7",  "P5",  "P3",  "P1",  "Pz",  "P2",  "P4",  "P6",  "P8",
    # Parieto-occipital
    "PO7", "PO3", "POz", "PO4", "PO8",
    # Occipital
    "O1",  "Oz",  "O2",
]

# ─── Device configurations ────────────────────────────────────────────────────
#
# train_device: True  → Device A (ZUNA trained on this layout)
# train_device: False → Devices B/C (zero-shot cross-device generalisation test)

DEVICE_CONFIGS: Dict[str, Dict] = {
    "emotiv_epoc": {
        "description": "Emotiv EPOC — 4 frontal channels",
        "input_channels": ["AF3", "AF4", "F3", "F4"],
        "train_device": True,          # Device A
        # C3/C4/P3/P4 are absent from this device → evaluate reconstruction there
        "eval_targets": ["C3", "C4", "P3", "P4"],
    },
    "muse_s": {
        "description": "Muse S — 4 temporal-frontal channels",
        "input_channels": ["AF7", "AF8", "T9", "T10"],
        "train_device": False,         # Device B
        "note": (
            "T9/T10 are the closest EEGMMIDB channels to Muse's TP9/TP10. "
            "TP9/TP10 are not in the standard 10-10 montage used by EEGMMIDB."
        ),
        # C3/C4/P3/P4 are absent from Muse S → same targets as Device A
        "eval_targets": ["C3", "C4", "P3", "P4"],
    },
    "openbci_cyton": {
        "description": "OpenBCI Cyton — 6 sensorimotor channels",
        "input_channels": ["C3", "C4", "P3", "P4", "Fz", "Cz"],
        "train_device": False,         # Device C
        # C3/C4/P3/P4 are INPUTS here — evaluate frontal/temporal channels instead
        # T7/T8: temporal; FC5/FC6: fronto-central — both absent from OpenBCI layout
        "eval_targets": ["T7", "T8", "FC5", "FC6"],
    },
}

# Month-1 reconstruction targets
INITIAL_TARGETS: List[str] = ["C3", "C4", "P3", "P4"]

# Motor-imagery run indices (PhysioNet 1-indexed)
# Runs 5/9/13: left-hand vs right-hand imagery (T1 vs T2)
# Runs 6/10/14: both-hands vs both-feet imagery (T1 vs T2)
MI_RUNS: List[int] = [5, 6, 9, 10, 13, 14]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_bad_channels(device_name: str, all_channels: List[str] = None) -> List[str]:
    """
    Return the list of channels to pass as bad_channels to ZUNA preprocessing.

    These are all channels in the recording EXCEPT the device's input channels.
    ZUNA zeroes these out before inference and then reconstructs them.

    Args:
        device_name: Key in DEVICE_CONFIGS.
        all_channels: Full channel list from the loaded .fif file.
                      Defaults to PHYSIONET_64_CHANNELS.

    Returns:
        List of channel names to mark as bad (i.e. to be reconstructed).
    """
    if all_channels is None:
        all_channels = PHYSIONET_64_CHANNELS

    input_set = set(DEVICE_CONFIGS[device_name]["input_channels"])
    return [ch for ch in all_channels if ch not in input_set]


def get_input_channels(device_name: str) -> List[str]:
    """Return the input (known) channels for a given device."""
    return DEVICE_CONFIGS[device_name]["input_channels"]


def list_devices() -> List[str]:
    """Return all device names."""
    return list(DEVICE_CONFIGS.keys())


def validate_channels_in_recording(
    device_name: str, recording_channels: List[str]
) -> None:
    """
    Assert that all input channels for a device exist in the recording.
    Raises ValueError listing any missing channels.
    """
    input_chs = set(DEVICE_CONFIGS[device_name]["input_channels"])
    recording_set = set(recording_channels)
    missing = input_chs - recording_set
    if missing:
        raise ValueError(
            f"Device '{device_name}' requires channels {sorted(missing)} "
            f"which are not present in the recording.\n"
            f"Recording has: {sorted(recording_channels)}"
        )
