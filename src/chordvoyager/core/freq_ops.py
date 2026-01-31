import math


def midi_freq(midi_value: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440 * (2 ** (float(midi_value - 69) / 12))


# https://arxiv.org/pdf/1306.6458#subsection.3.2
def approximate_frequency_ratio(
    f1: float, f2: float, d: float = 0.01, divide_by: int = 1
) -> tuple[int, int, float]:
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


def relative_periodicity(
    semitones: list[int], reference_index: int = 0, d: float = 0.011, sort: bool = True
) -> int:
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
        relative_periodicities.append(
            relative_periodicity(current_semitones, reference_index=i, d=d)
        )
    if log:
        smoothed_periodicity = sum([math.log2(rp) for rp in relative_periodicities]) / n
    else:
        smoothed_periodicity = sum(relative_periodicities) / n
    if verbose:
        print(f"Chord {semitones}, dissonance: {smoothed_periodicity}")

    return smoothed_periodicity


def midi_interval_ratio(midi1: int, midi2: int, d: float = 0.011) -> tuple[int, int, float]:
    f1 = midi_freq(midi1)
    f2 = midi_freq(midi2)
    return approximate_frequency_ratio(f1, f2, d)


def dissonance(semitones: list[int]):
    return smoothed_relative_periodicity(semitones, verbose=False)
