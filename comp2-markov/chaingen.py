import itertools

possible_chords = ["I", "i", "II", "ii", "III", "iii", "IV", "iv", "V", "v", "VI", "vi", "VII", "vii"]

substitution_table = {}

for state in itertools.product(possible_chords, repeat=3):
    substitution_distribution = {}
    chord_default = state[2]
    idx_0 = (((possible_chords.index(state[0]) // 2) + 3) * 2) % len(possible_chords)
    valid_degrees = [idx_0, idx_0 + 1]
    idx_1 = possible_chords.index(state[1])
    # if 2nd chord's index is 3 degrees higher
    if idx_1 in valid_degrees:
        target_chord = possible_chords[(idx_1 + 6) % len(possible_chords)]
        target_weight = 0.95
        remaining_weight = 1 - target_weight
        for chord in possible_chords:
            if chord == target_chord:
                substitution_distribution[chord] = target_weight
            else:
                substitution_distribution[chord] = remaining_weight / (len(possible_chords) - 1)
    else:
        default_prob = 0.85
        other_prob = (1 - default_prob) / (len(possible_chords) - 1)
        for chord in possible_chords:
            if chord == chord_default:
                substitution_distribution[chord] = default_prob
            else:
                substitution_distribution[chord] = other_prob
    substitution_table[state] = substitution_distribution

from random import choice

print("Total number of states:", len(substitution_table))
print("\nSome sample states and their substitution distributions:\n")
sample_states = [choice(list(substitution_table.keys())) for _ in range(3)]
for s in sample_states:
    print(f"State {s}:")
    for chord, prob in substitution_table[s].items():
        print(f"   {chord}: {prob:.3f}")
    print()
