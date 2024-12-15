import pretty_midi
import math
from itertools import chain, combinations
from music21 import converter
import os
import numpy as np

# Convert a midi note to its corresponding pitch class
def midi_to_pitch_class(midi_note):
    return midi_note % 12

# Given a midi file, return all the notesets over a duration, default 1 second
def get_notesets_by_time(midi_data, segment_duration=1.0):
    instrument = midi_data.instruments[0]        
    notes = instrument.notes
    max_time = max(note.end for note in notes)
    
    # Time-segmented notesets
    notesets = []
    current_time = 0
    
    while current_time < max_time:
        segment_noteset = set(
            midi_to_pitch_class(note.pitch) 
            for note in notes 
            if current_time <= note.start < current_time + segment_duration
        )
        notesets.append(segment_noteset)
        current_time += segment_duration
    
    return notesets

# Down-scale a noteset just like the paper suggests
def half_resolution(time_notesets):
    half_resolution = [None for i in range(int(len(time_notesets) / 2))]
    for i in range(1, int(len(time_notesets) / 2)):
        half_resolution[i] = time_notesets[2*i - 1].union(time_notesets[2*i])
    
    return half_resolution[1:]

# Find a specific note's place on a tonic note's chromatic scale as defined in the paper
class chromatic_scale:
    def __init__(self):
        self.tone_to_graph = { # circle then scale
            9  : (["D#", "A#", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#"], ["D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D"]),
            10 : (["E", "B", "F#", "C#", "G#", "D#", "A#", "F", "C", "G", "D", "A"], ["E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#"]),
            11 : (["F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"], ["F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E"]),
            0  : (["F#", "C#", "G#", "D#", "A#", "F", "C", "G", "D", "A", "E", "B"], ["F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F"]),
            1  : (["G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F", "C"], ["G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#"]),
            2  : (["G#", "D#", "A#", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"], ["G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G"]),
            3  : (["A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F", "C", "G", "D"], ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]),
            4  : (["A#", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#"], ["A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A"]),
            5  : (["B", "F#", "C#", "G#", "D#", "A#", "F", "C", "G", "D", "A", "E"], ["B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#"]),
            6  : (["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"], ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
            7  : (["C#", "G#", "D#", "A#", "F", "C", "G", "D", "A", "E", "B", "F#"], ["C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]),
            8  : (["D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F", "C", "G"], ["D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#"])
        }
                
        self.note_to_pitch_class = {
            3  : "D#",
            10 : "A#",
            5  : "F",
            0  : "C",
            7  : "G",
            2  : "D",
            9  : "A",
            4  : "E",
            11 : "B",
            6  : "F#",
            1  : "C#",
            8  : "G#"
        }
        
    def note_to_coor(self, note, tone):
        x_axis, y_axis = self.tone_to_graph[tone]
        return x_axis.index(self.note_to_pitch_class[note]) - 6, y_axis.index(self.note_to_pitch_class[note]) - 6

# Generate the harmonic point of a noteset around a given tonic note
def generate_harmonic_point(set, chrom, tone):
    h_x = 0
    h_y = 0
    for n in set:
        x,y = chrom.note_to_coor(n, tone)
        h_x += x
        h_y += y
        
    return h_x, h_y

# Find the center offset of a noteset as defined in the heuristics paper
def center_offset(noteset, chrom):
    result = []
    if len(noteset) == 0:
        return 0
    for tone in noteset:
        s = list(noteset)
        sets = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
        geo_x = 0
        geo_y = 0
        for set in sets:
            x, y = generate_harmonic_point(set, chrom, tone)
            geo_x += x
            geo_y += y
        geo_x /= len(sets)
        geo_y /= len(sets)
        
        result.append(math.sqrt((geo_x**2.0) + (geo_y**2.0)))
    return max(result) - min(result)



# H1 Heuristic
def get_h1(time_notesets):
    result_sum = 0
    half_set = time_notesets
    for i in range(0, int(math.log(len(time_notesets)))):
        denominator = sum([len(x) for x in half_set]) / len(half_set)
        half_set    = half_resolution(half_set)
        numerator   = (sum([len(x) for x in half_set]) / len(half_set))  if len(half_set) != 0 else 0
        result_sum += (numerator / denominator) if denominator != 0 else 0
    return result_sum / math.log(len(time_notesets))

# H2 Heuristic
def get_h2(time_notesets):
    chrom = chromatic_scale()
    result_sum = 0

    half_set = time_notesets
    
    cos = [center_offset(ns, chrom) for ns in half_set] # For every set in downgraded music, which is 
    for i in range(0, int(math.log(len(time_notesets)))): #Must check if this bound is right
        denominator = sum(cos) / len(half_set)
        half_set    = half_resolution(half_set)
        cos = [center_offset(x, chrom) for x in half_set]
        numerator   = (sum(cos) / len(half_set))  if len(half_set) != 0 else 0
        result_sum += (numerator / denominator) if denominator != 0 else 0
    return result_sum / math.log(len(time_notesets))

# H3 Heuristic
def get_h3(time_notesets):
    result_sum = 0
    half_set = time_notesets
    for i in range(0, int(math.log(len(time_notesets)))): #Must check if this bound is right
        denominator = max([len(half_set[x+1]) - len(half_set[x]) for x in range(0, len(half_set)-2)])
        half_set    = half_resolution(half_set)
        numerator   = max([len(half_set[x+1]) - len(half_set[x])  for x in range(0, max(1, len(half_set)-2) )]) if len(half_set) > 1 else 0
        result_sum += (numerator / denominator) if denominator != 0 else 0
    return result_sum / math.log(len(time_notesets))

# H4 Heuristic
def get_h4(h1, h2, h3):
    return h1 * h2 * h3



def main():

    dir_path = '/path/to/midi_directory'



    h1 = []
    h2 = []
    h3 = []
    h4 = []

    for i in os.listdir(dir_path):
        # Load the MIDI file
        song_string = '{}/{}'.format(dir_path, i)
        
        midi_data = None 
        try:
            midi_data = pretty_midi.PrettyMIDI(song_string)
        except:
            print("The following file couldn't be read, skipping:", i)
            continue

        # Example: Notesets segmented by 1-second intervals
        time_notesets = get_notesets_by_time(midi_data, segment_duration=2.0)
        time_notesets = [x for x in time_notesets if len(x) != 0]

        song_h1 = get_h1(time_notesets)
        song_h2 = get_h2(time_notesets)
        song_h3 = get_h3(time_notesets)
        song_h4 = get_h4(song_h1, song_h2, song_h3)

        h1.append(song_h1)
        h2.append(song_h2)
        h3.append(song_h3)
        h4.append(song_h4)
    
    print("H1:", np.mean(h1))
    print("H2:", np.mean(h2))
    print("H3:", np.mean(h3))
    print("H4:", np.mean(h4))



if __name__ == '__main__':
    main()
