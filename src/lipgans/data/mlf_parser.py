from pathlib import Path
from typing import List, Tuple, Dict

# HTK MLF times are in 100 ns units
TIME_SCALE = 1e7


def parse_mlf(mlf_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Parse an HTK MLF file into a dictionary of phoneme timings.

    Args:
        mlf_path (Path): Path to the .mlf file

    Returns:
        Dict[str, List[Tuple[float, float, str]]]:
            {
                "record_name": [(start_s, end_s, phoneme), ...],
                ...
            }
    """
    lines = Path(mlf_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    records: Dict[str, List[Tuple[float, float, str]]] = {}
    current_rec = None
    out: List[Tuple[float, float, str]] = []

    for line in lines:
        line = line.strip()

        if line.startswith('"'):  # Start of new record
            if current_rec and out:  # save previous
                records[current_rec] = out
            current_rec = Path(line.strip('"')).stem  # use filename stem as key
            out = []
            continue

        if line == ".":  # End of record
            if current_rec and out:
                records[current_rec] = out
            current_rec, out = None, []
            continue

        parts = line.split()
        if len(parts) == 3:
            try:
                start, end = int(parts[0]), int(parts[1])
                phon = parts[2].lower()

                if phon == "sil":  # Skip silence
                    continue

                out.append((start / TIME_SCALE, end / TIME_SCALE, phon))

            except ValueError:
                continue

    return records


# Example Usage
# records = parse_mlf(Path("aligned.mlf"))

# print(len(records))  
# # e.g., 1342 records

# print(records["s001_001"][:5])  
# # [(0.0, 0.12, 'hh'), (0.12, 0.20, 'ah'), (0.20, 0.35, 'l'), (0.35, 0.50, 'ow')]
