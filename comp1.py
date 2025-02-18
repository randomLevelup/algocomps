"""

comp1.py
by jupiter westbard

"""

import random, argparse

from music21 import environment, stream, scale, note, roman
# us = environment.UserSettings()
# us['musescoreDirectPNGPath'] = "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"
# us['musicxmlPath'] = "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"

def get_context(c_string, idx):
    """Returns tuple of (left, current, right) context characters at index idx in c_string."""
    env_context = '1'
    ignore = ['[', ']', '+', '-', 'F']

    # find left context
    left_context = env_context
    i = idx
    while i > 0:
        if c_string[i-1] in ['+', '-', 'F']:   # left neighbor is not valid context
            i -= 1
        else:
            if c_string[i-1] == '[':          # left neighbor is an opening bracket
                depth = 0
                j = i - 2
                while j >= 0 and (depth != 0 or c_string[j] in ignore):
                    if c_string[j] == ']':
                        depth += 1
                    elif c_string[j] == '[':
                        depth -= 1
                    j -= 1
                if j >= 0:
                    left_context = c_string[j]
            elif c_string[i-1] == ']':        # left neighbor is a closing bracket
                depth = -1
                j = i - 2
                while j >= 0 and (depth != 0 or c_string[j] in ignore):
                    if c_string[j] == '[':
                        depth += 1
                    elif c_string[j] == ']':
                        depth -= 1
                    j -= 1
                if j >= 0:
                    left_context = c_string[j]
            else:                             # left neighbor is a normal character
                left_context = c_string[i-1]
            break
    
    # find right context
    right_context = env_context
    i = idx
    while i < len(c_string) - 1:
        if c_string[i+1] in ['+', '-', 'F']:       # right neighbor is not valid context
            i += 1
        else:
            if c_string[i+1] == ']':          # right neighbor is a closing bracket
                right_context = env_context
            elif c_string[i+1] == '[':        # right neighbor is an opening bracket
                depth = 1
                j = i + 2
                while j < len(c_string) and (depth != 0 or c_string[j] in ignore):
                    if c_string[j] == '[':
                        depth += 1
                    elif c_string[j] == ']':
                        depth -= 1
                    j += 1
                if j < len(c_string):
                    right_context = c_string[j]
            else:                             # right neighbor is a normal character
                right_context = c_string[i+1]
            break
    
    return left_context, c_string[idx], right_context

def gen_lsystem_contextual(axiom, rules, iterations):
    """Generates L-system string by applying contextual rules to axiom for given iterations."""
    current = axiom
    for _ in range(iterations):
        next_str = []
        for i, char in enumerate(current):
            if char not in ['[', ']', '+', '-', 'F']: # char is context-dependent
                context = get_context(current, i)
                if context in rules:
                    next_str.append(rules[context])
                else:
                    print(f'!!! No rule for {context}')
            elif char in ['+', '-']:                  # char is a pitch modifier
                next_str.append(rules[char])
            else:                                     # char is unchanged
                next_str.append(char)
        current = ''.join(next_str)
    return current

def generate_music_contextual(lsystem_string):
    """Converts L-system string to music21 Stream using scale-based note generation."""
    s1 = stream.Stream()
    F_major = scale.WeightedHexatonicBlues('G')
    scale_notes = F_major.getPitches('G2', 'G6')
    current_note_idx = len(scale_notes) // 2  # Start in middle of range
    current_duration = 0
    stack = []

    # Process each character in the L-system string
    for char in lsystem_string:
        if char == 'F':
            current_duration += 1.0  # Add quarter note duration
        elif char == '+':
            current_note_idx = (current_note_idx + 1) % (len(scale_notes) - 1)
        elif char == '-':
            current_note_idx = abs(1 - current_note_idx) % (len(scale_notes) - 1)
        elif char == '[':
            stack.append((current_note_idx, current_duration))
            current_duration = 0
        elif char == ']':
            if current_duration > 0:
                n = note.Note(scale_notes[current_note_idx], quarterLength=current_duration)
                s1.append(n)
            current_note_idx, current_duration = stack.pop()

    return s1


'''
If you want to see the visualization of the plant, install jupyturtle,
run the following code in a jupyter notebook, run __main__ until the
draw_plant_contextual function is called.
'''
# from jupyturtle import *

# def draw_plant_contextual(instructions, angle, length, length_decay=0.8):
#     make_turtle(delay=0, width=400, height=400)
#     set_heading(270)
#     jp(200, 380)
#     set_color("gray")
#     hide()

#     stack = []
#     last_cmd = '1'
#     for cmd in instructions:
#         if cmd == 'F':
#             fd(length)
#         elif cmd in ['1', '0']:
#             last_cmd = cmd
#         elif cmd == '[':
#             t = get_turtle()
#             stack.append((t.position, t.heading, length))
#             direction = 1 if last_cmd == '1' else -1
#             set_heading(t.heading + (angle * direction))
#             length *= length_decay
#         elif cmd == ']':
#             pos, head, length = stack.pop()
#             jp(pos.x, pos.y)
#             set_heading(head)
    

roman_to_degree = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4,
    'V': 5, 'VI': 6, 'VII': 7,
}

degree_to_roman = {
    1: 'I', 2: 'II', 3: 'III', 4: 'IV',
    5: 'V', 6: 'VI', 7: 'VII'
}

def get_degree(tonic):
    modifier = 0
    while tonic.startswith('b'):
        tonic = tonic[1:]
        modifier -= 1
    while tonic.startswith('#'):
        tonic = tonic[1:]
        modifier += 1
    return roman_to_degree[tonic] + modifier, modifier

def get_roman(degree, modifier):
    out = degree_to_roman[degree]
    while modifier < 0:
        out = 'b' + out
        modifier += 1
    while modifier > 0:
        out = '#' + out
        modifier -= 1
    return out

def get_subdominant(tonic):
    degree, modifier = get_degree(tonic)
    subdominant_degree = (degree - 1 + 3) % 7 + 1 # 4th degree relative to tonic
    return get_roman(subdominant_degree, modifier)

def get_dominant(tonic):
    degree, modifier = get_degree(tonic)
    dominant_degree = (degree - 1 + 4) % 7 + 1 # 5th degree relative to tonic
    return get_roman(dominant_degree, modifier)

def get_supertonic(tonic):
    degree, modifier = get_degree(tonic)
    supertonic_degree = (degree - 1 + 1) % 7 + 1 # 2nd degree relative to tonic
    return get_roman(supertonic_degree, modifier)

def get_flat_supertonic(tonic):
    degree, modifier = get_degree(tonic)
    supertonic_degree = (degree - 1 + 1) % 7 + 1 # 2nd degree relative to tonic
    return get_roman(supertonic_degree, modifier - 1) # flatten

def get_mediant(tonic):
    degree, modifier = get_degree(tonic)
    mediant_degree = (degree - 1 + 5) % 7 + 1 # 6th degree relative to tonic
    return get_roman(mediant_degree, modifier)


class Terminal:
    def __init__(self, root, isMinor, isDom7, sD, dom, sT, flatST, mD, duration):
        self.duration = duration # in quarter notes
        self.root = root         # root chord
        self.isMinor = isMinor   # default: 'maybe'
        self.isDom7 = isDom7     # default: 'maybe'
        self.sD = sD             # chord rooted at subdominant of x
        self.dom = dom           # chord at the dominant of x
        self.sT = sT             # supertonic of x
        self.flatST = flatST     # flat supertonic of x
        self.mD = mD             # mediant of x
        self.rootHasChanged = False
    
    def print(self):
        out = self.root
        out += 'm' if self.isMinor == 'yes' else ''
        out += '7' if self.isDom7 == 'yes' else ''
        out = ("SubDom " if self.sD else "") + out
        out = ("Dom " if self.dom else "") + out
        out = ("SupTonic " if self.sT else "") + out
        out = ("FlatSupTonic " if self.flatST else "") + out
        out = ("Mediant " if self.mD else "") + out
        out += f" ({self.duration})"
        return out

def tm(root, isMinor='maybe', isDom7='maybe',
       sD=False, dom=False, sT=False, flatST=False, mD=False,
       duration=-1):
    return Terminal(root, isMinor, isDom7, sD, dom, sT, flatST, mD, duration)

def equal_terms(t1, t2):
    out = t2.root == 'x' or t2.root == 'w' or t1.root == t2.root
    out = out and (t1.isMinor == 'maybe' or t2.isMinor == 'maybe' or t1.isMinor == t2.isMinor)
    out = out and (t1.isDom7 == 'maybe' or t2.isDom7 == 'maybe' or t1.isDom7 == t2.isDom7)
    out = out and t1.sD == t2.sD
    out = out and t1.dom == t2.dom
    out = out and t1.sT == t2.sT
    out = out and t1.flatST == t2.flatST
    out = out and t1.mD == t2.mD
    return out

class Rule:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

def apply_transformations(term_root, rule_rhs_term):
    """Apply root transformations based on rule terms."""
    if rule_rhs_term.sD:
        return get_subdominant(term_root), True
    elif rule_rhs_term.dom:
        return get_dominant(term_root), True
    elif rule_rhs_term.sT:
        return get_supertonic(term_root), True
    elif rule_rhs_term.flatST:
        return get_flat_supertonic(term_root), True
    elif rule_rhs_term.mD:
        return get_mediant(term_root), True
    return term_root, False

def determine_property(rule_term, orig_term, property_name):
    """Determine minor/dom7 property based on rule and original term."""
    rule_value = getattr(rule_term, property_name)
    orig_value = getattr(orig_term, property_name)
    if rule_value == 'yes' or orig_value == 'yes':
        return 'yes'
    elif rule_value == 'no' or orig_value == 'no':
        return 'no'
    return 'maybe'

chord_rule_prob = 0.25

def apply_single_term_rule(terms, rule):
    """Handle rules with single term on LHS."""
    new_terms = []
    rule_lhs = rule.lhs[0]
    rhs_durations = None
    
    for term in terms:
        if not equal_terms(term, rule_lhs):
            new_terms.append(term)
            continue
            
        if term.root != 'S12' and random.random() >= chord_rule_prob:
            new_terms.append(term)
            continue
            
        print(f"applying to {term.root} with duration {term.duration}")
        rhs_durations = term.duration // len(rule.rhs)
        
        for rule_rhs_term in rule.rhs:
            rhs_term_root = rule_rhs_term.root if rule_rhs_term.root != 'x' else term.root
            rhs_term_root, rootChange = apply_transformations(rhs_term_root, rule_rhs_term)
            
            termIsMinor = determine_property(rule_rhs_term, term, 'isMinor')
            termIsDom7 = determine_property(rule_rhs_term, term, 'isDom7')
            
            new_term = tm(rhs_term_root,
                         isMinor=termIsMinor,
                         isDom7=termIsDom7,
                         duration=rhs_durations)
            if rootChange:
                new_term.rootHasChanged = True
            new_terms.append(new_term)
            
    return new_terms

def apply_multi_term_rule(terms, rule):
    """Handle rules with multiple terms on LHS."""
    new_terms = []
    i = 0
    
    while i < len(terms):
        if i + len(rule.lhs) > len(terms):
            new_terms.append(terms[i])
            i += 1
            continue
            
        matches = True
        for j in range(len(rule.lhs)):
            if not equal_terms(terms[i + j], rule.lhs[j]):
                matches = False
                break
            if rule.lhs[0].root == 'w' and (terms[i].dom or terms[i].rootHasChanged):
                matches = False
                break
                
        if matches and random.random() < chord_rule_prob:
            print(f"applying to sequence starting at {terms[i].root}")
            for j, rule_rhs_term in enumerate(rule.rhs):
                rhs_term_root = terms[i].root if rule_rhs_term.root in ['x', 'w'] else rule_rhs_term.root
                rhs_term_root, rootChange = apply_transformations(rhs_term_root, rule_rhs_term)
                
                termIsMinor = determine_property(rule_rhs_term, terms[i], 'isMinor')
                termIsDom7 = determine_property(rule_rhs_term, terms[i], 'isDom7')
                
                new_term = tm(rhs_term_root,
                             isMinor=termIsMinor,
                             isDom7=termIsDom7, 
                             duration=terms[i+j].duration)
                if rootChange:
                    new_term.rootHasChanged = True
                new_terms.append(new_term)
            i += len(rule.lhs)
        else:
            new_terms.append(terms[i])
            i += 1
            
    return new_terms

def apply_rule(terms, rule):
    """Apply transformation rules to terms."""
    if len(rule.lhs) == 1:
        return apply_single_term_rule(terms, rule)
    return apply_multi_term_rule(terms, rule)

def generate_seed(num_iters, num_rules):
    out = ""
    for _ in range(num_iters):
        out += str(random.randint(1, num_rules))
    return out

def create_chord_from_term(term):
    """Converts a Terminal object to a music21 RomanNumeral chord with appropriate duration."""
    term_root = term.root

    if term.isMinor == 'yes':
        term_root = term_root.lower()

    if term.isDom7 == 'yes':
        term_root += '7'

    # Convert roman numeral to chord
    root = roman.RomanNumeral(term_root, 'G')
    
    # Set duration
    root.quarterLength = term.duration
    return root


chord_rules_library = [
    Rule([tm('x')], [tm('x', isDom7='no'), tm('x')]),
    Rule([tm('x')], [(tm('x')), tm('x', sD=True)]),
    Rule([tm('w'), tm('x', isMinor='no',  isDom7='yes')], [tm('x', dom=True, isDom7='yes'), tm('x')]),
    Rule([tm('w'), tm('x', isMinor='yes', isDom7='yes')], [tm('x', dom=True, isMinor='no'),  tm('x')]),
]

chord_rule0 = Rule([tm('S12', duration=4*12)], [
    tm('I', isDom7='no'),                 # I(m)
    tm('I', isMinor='no', isDom7='yes'),  # I7
    tm('IV', isDom7='no'),                # IV(m)
    tm('I', isDom7='no'),                 # I(m)
    tm('V', isMinor='no', isDom7='yes'),  # V7
    tm('I', isDom7='no')                  # I(m)
])

chord_rule_prob = 0.25

melody_rules_library = [
    ['0', '1[+F1]', '1', '1', '0', '1F1', '0', '0', '-', '+'],
    ['1', '1[-F1]', '1', '1', '0', '1F1', '1', '0', '-', '+'],
    ['0', '1', '0', '1[+F1]', '0', '1F1', '0', '0', '-', '+'],
    ['1', '0', '0', '1F1', '1', '1[+F1]', '1', '0', '-', '+'],
    ['0', '1[+F1]', '1', '1', '0', '1F1', '1', '0', '-', '+'],
]

melody_axioms_library = ['F0F1', 'F1F0', 'F1F1']

plant_config = {
    'axiom': random.choice(melody_axioms_library),
    'rules': {l: r for l, r in zip([
            ('0', '0', '0'),
            ('0', '0', '1'),
            ('0', '1', '0'),
            ('0', '1', '1'),
            ('1', '0', '0'),
            ('1', '0', '1'),
            ('1', '1', '0'),
            ('1', '1', '1'),
            '+',
            '-'],
            random.choice(melody_rules_library))},
    'iterations': 22,
    'angle': 25,
    'length': 10,
    'thickness': 5,
    'length_decay': 0.9
}

def main(output_path=None, show_midi=False):
    # generate melody
    print("generating melody...")
    print(f"axiom: {plant_config['axiom']}")
    gen_string_length = 9999
    while gen_string_length > 1000:
        generated_string = gen_lsystem_contextual(plant_config['axiom'],
                                                plant_config['rules'],
                                                plant_config['iterations'])
        gen_string_length = len(generated_string)
        plant_config['iterations'] -= 1

    print(generated_string)    

    s1 = generate_music_contextual(generated_string)

    # slash the offsets by 4x (effectively increasing the tempo)
    for element in s1.recurse():
        if hasattr(element, 'duration'):
            element.duration.quarterLength /= 1
        if hasattr(element, 'offset'):
            element.offset /= 4.0
    
    # generate chords
    print("\ngenerating chords...")
    chord_iterations = 5
    chord_seed = generate_seed(chord_iterations, len(chord_rules_library))
    print(f"seed: [{chord_seed}]\n")

    axiom = [tm('S12', isMinor='yes', duration=4*12)]
    axiom = apply_rule(axiom, chord_rule0)
    print("axiom S12 -->")
    print (' | '.join([term.print() for term in axiom]))

    for i in range(chord_iterations):
        pick_rule = int(chord_seed[i])

        if pick_rule == 3:
            print(f"rule 3a")
        elif pick_rule == 4:
            print(f"rule 3b")
        elif pick_rule > 4:
            print(f"rule {pick_rule - 1}")
        else:
            print(f"rule {pick_rule}")

        axiom = apply_rule(axiom, chord_rules_library[pick_rule - 1])
        print(' | '.join([term.print() for term in axiom]))

    s2 = stream.Stream()
    total_s1_duration = sum([n.quarterLength for n in s1.recurse().notes])
    current_s2_duration = sum([c.quarterLength for c in s2.recurse().notes])

    while current_s2_duration < total_s1_duration + 8:
        for term in axiom:
            chord = create_chord_from_term(term)
            chord.inversion(random.choice([0, 1, 2]))
            chord.transpose(-12, inPlace=True)
            s2.append(chord)
            current_s2_duration += chord.quarterLength * 4

            if current_s2_duration >= total_s1_duration + 8:
                break

    s3 = stream.Score()
    s3.insert(0, s1)
    s3.insert(0, s2)

    # Write to specified output path or default
    output_file = output_path if output_path else 'comp1.mid'
    s3.write('midi', output_file)
    
    if show_midi:
        s3.show('midi')

    return 0

def __main__():
    parser = argparse.ArgumentParser(description='Generate algorithmic music composition')
    parser.add_argument('--showmidi', action='store_true', help='Show MIDI output')
    parser.add_argument('output', nargs='?', help='Output MIDI file path', default=None)
    
    args = parser.parse_args()
    return main(args.output, args.showmidi)

if __name__ == '__main__':
    __main__()