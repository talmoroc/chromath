import math


def midi_freq(midi_value: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440 * (2 ** (float(midi_value - 69) / 12))


# https://arxiv.org/pdf/1306.6458#subsection.3.2
def approximate_frequency_ratio(f1: float, f2: float, d: float = 0.01, divide_by: int = 1) -> tuple[int, int, float]:
    r_init = f2 / f1
    r_min, r_max = (r_init * (1 - d), r_init * (1 + d))
    a_low, b_low = math.floor(r_init), 1
    a_high, b_high = math.ceil(r_init), 1
    a, b = round(r_init), 1
    r = a / b

    while r < r_min or r > r_max:
        r0 = 2 * r_init - r
        if r_init < r:
            a_high, b_high = a, b
            k = math.floor((r0 * b_low - a_low) / (a_high - r0 * b_high))
            a_low, b_low = (a_low + k * a_high, b_low + k * b_high)
        else:
            a_low, b_low = a, b
            k = math.floor((a_high - r0 * b_high) / (r0 * b_low - a_low))
            a_high, b_high = (a_high + k * a_low, b_high + k * b_low)
        a, b = a_low + a_high, b_low + b_high
        r = a / b
    b = b * divide_by
    r = r / divide_by
    return (a, b, r)


def relative_periodicity(semitones: list[int], reference_index: int = 0, d: float = 0.011, sort: bool = True) -> int:
    reference_freq = 2 ** (semitones[reference_index] / 12)
    ratios = [
        approximate_frequency_ratio(
            reference_freq,
            2 ** ((s + 12 * (i < reference_index)) / 12),
            d,
            divide_by=(i < reference_index) + 1,
        )
        for i, s in enumerate(semitones)
    ]
    denominators = [r[1] for r in ratios]
    return int(ratios[0][2] * math.lcm(*denominators))


def smoothed_relative_periodicity(
    semitones: list[int],
    d: float = 0.011,
    sort: bool = True,
    log: bool = True,
    verbose: bool = True,
) -> float:
    current_semitones = sorted(list(set(semitones))) if sort else list(set(semitones))
    n = len(current_semitones)
    relative_periodicities: list[int] = []
    for i in range(len(current_semitones)):
        current_semitones = [s - current_semitones[i] for s in current_semitones]
        relative_periodicities.append(relative_periodicity(current_semitones, reference_index=i, d=d))
    smoothed_periodicity = sum([math.log2(rp) for rp in relative_periodicities]) / n if log else sum(relative_periodicities) / n
    if verbose:
        print(f"Chord {semitones}, dissonance: {smoothed_periodicity}")

    return smoothed_periodicity


def midi_interval_ratio(midi1: int, midi2: int, d: float = 0.011) -> tuple[int, int, float]:
    f1 = midi_freq(midi1)
    f2 = midi_freq(midi2)
    return approximate_frequency_ratio(f1, f2, d)


def relative_dissonance(semitones: list[int], d: float = 0.011, reference_index: int = 0) -> float:
    """Dissonance based on single-reference relative periodicity (not smoothed).

    Computes log2 of the relative periodicity with respect to a single
    reference note (by default, the lowest/first note). No averaging
    across multiple reference points. Provides a raw measure of how
    complex the chord is relative to one fundamental.
    """
    unique = sorted(set(semitones))
    if len(unique) < 1:
        return 0.0
    # Normalize semitones relative to the reference
    normalized = [s - unique[reference_index] for s in unique]
    rp = relative_periodicity(normalized, reference_index=0, d=d)
    return math.log2(rp)


def pairwise_dissonance(semitones: list[int], d: float = 0.011) -> float:
    """Sum of log2(relative_periodicity) for every pair of notes.

    More sensitive to individual clashing intervals than
    ``smoothed_relative_periodicity``, which averages across reference
    notes.  A minor triad scores lower than a major-7th chord because
    no pair in a minor triad has a complex ratio.
    """
    unique = sorted(set(semitones))
    if len(unique) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            rp = relative_periodicity([unique[i], unique[j]], reference_index=0, d=d)
            total += math.log2(rp)
            count += 1
    return total / count if count else 0.0


def max_pairwise_dissonance(semitones: list[int], d: float = 0.011) -> float:
    """Maximum pairwise dissonance across all note pairs.

    Captures the single worst interval clash in a chord.  Useful for
    penalising chords that contain one very dissonant interval even if
    the rest are consonant.
    """
    unique = sorted(set(semitones))
    if len(unique) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            rp = relative_periodicity([unique[i], unique[j]], reference_index=0, d=d)
            worst = max(worst, math.log2(rp))
    return worst


# Registry of dissonance methods
DISSONANCE_METHODS: dict[str, type[None]] = {}  # populated below


def dissonance(
    semitones: list[int],
    method: str = "smoothed",
) -> float:
    """Compute dissonance of a chord using the chosen method.

    Available methods:
        - ``"smoothed"`` (default): log2 of smoothed relative periodicity
          across all reference notes.
        - ``"smoothed_raw"``: raw (linear) smoothed relative periodicity,
          without log2 transform.  Larger absolute values but may be more
          sensitive to consonance/dissonance distinctions for triads.
        - ``"relative"``: log2 of relative periodicity with respect to
          lowest note, not smoothed. Raw complexity measure.
        - ``"pairwise"``: mean log2(relative_periodicity) over all note
          pairs.  Better at distinguishing simple triads from complex
          chords.
        - ``"max_pairwise"``: maximum single-pair dissonance.  Penalises
          the worst interval clash.
    """
    if method == "smoothed":
        return smoothed_relative_periodicity(semitones, verbose=False, log=True)
    elif method == "smoothed_raw":
        return smoothed_relative_periodicity(semitones, verbose=False, log=False)
    elif method == "relative":
        return relative_dissonance(semitones)
    elif method == "raw":
        return relative_periodicity(semitones)
    elif method == "pairwise":
        return pairwise_dissonance(semitones)
    elif method == "max_pairwise":
        return max_pairwise_dissonance(semitones)
    else:
        raise ValueError(f"Unknown dissonance method {method!r}. Choose from: 'smoothed', 'smoothed_raw', 'relative', 'pairwise', 'max_pairwise'.")
