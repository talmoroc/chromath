import mido


# WIP
def find_port_name() -> str:
    for name in mido.get_output_names():  # type: ignore
        if "Python MIDI" in name:
            return name
    raise RuntimeError("MIDI output port 'Python MIDI' not found.")


def __main__():
    port_name = find_port_name()
    outport = mido.open_output(port_name)  # type: ignore  # noqa: F841


__main__()
