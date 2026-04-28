"""
Fixed 87/22 train/test subject split for PhysioNet EEGMMIDB.

Split is deterministic and must never change once the paper is written.
Subjects 1-87  → training set
Subjects 88-109 → test set

The split was chosen to be a clean contiguous partition so it is trivially
reproducible without storing any file.  Subject IDs follow PhysioNet convention
(1-indexed integers; files are named S001 … S109).
"""

from typing import List, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

N_SUBJECTS: int = 109
N_TRAIN:    int = 87
N_TEST:     int = 22   # 87 + 22 = 109

TRAIN_SUBJECTS: List[int] = list(range(1,  88))   # [1, 2, …, 87]
TEST_SUBJECTS:  List[int] = list(range(88, 110))  # [88, 89, …, 109]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_split() -> Tuple[List[int], List[int]]:
    """
    Return (train_subjects, test_subjects) as lists of integer subject IDs.

    Example
    -------
    >>> train, test = get_split()
    >>> len(train), len(test)
    (87, 22)
    >>> train[0], train[-1]
    (1, 87)
    >>> test[0], test[-1]
    (88, 109)
    """
    return TRAIN_SUBJECTS, TEST_SUBJECTS


def subject_id_to_str(subject_id: int) -> str:
    """
    Convert integer subject ID to zero-padded string (e.g. 1 → 'S001').
    Used to build file names and directory paths.
    """
    return f"S{subject_id:03d}"


def subject_str_to_id(subject_str: str) -> int:
    """
    Convert zero-padded string back to integer (e.g. 'S001' → 1).
    Accepts both 'S001' and '001'.
    """
    return int(subject_str.lstrip("S"))


def is_train(subject_id: int) -> bool:
    return subject_id in set(TRAIN_SUBJECTS)


def is_test(subject_id: int) -> bool:
    return subject_id in set(TEST_SUBJECTS)


if __name__ == "__main__":
    train, test = get_split()
    print(f"Train: {len(train)} subjects  ({train[0]}–{train[-1]})")
    print(f"Test : {len(test)} subjects  ({test[0]}–{test[-1]})")
    assert len(train) + len(test) == N_SUBJECTS
    assert len(set(train) & set(test)) == 0, "Train/test overlap!"
    print("Split OK.")
