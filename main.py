import mido

from engine_utils.computation_utils import *
from music_objects import *


def find_port_name() -> str:
    for name in mido.get_output_names():  # type: ignore
        if 'Python MIDI' in name:
            return name
    raise RuntimeError("MIDI output port 'Python MIDI' not found.")



def __main__():
    port_name = find_port_name()
    outport = mido.open_output(port_name)  # type: ignore

    A4 = Note(A4_MIDI_VALUE)
    c = Chord.from_shape(A4, ChordShape.m7)
    print(c)
    c2 = c.invert(1)
    print(c2)

    # test1 = Chord([0, 4, 7])
    # test = Chord.from_shape(2, ChordShape.MINOR)
    # test1.play(outport)
    # test.play(outport)
    # print(test)

__main__()
