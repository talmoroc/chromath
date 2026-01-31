import mido
import time
import threading


def _play_unthreaded(self, outport, duration: int = 500, velocity: int = 80):
    for n in self.midi:
        outport.send(mido.Message("note_on", note=n, velocity=velocity))
    time.sleep(duration / 1000)
    for n in self.midi:
        outport.send(mido.Message("note_off", note=n, velocity=velocity))


def play(self, duration: int = 500, velocity: int = 80):
    thread = threading.Thread(target=self._play_unthreaded, args=(duration, velocity))
    thread.start()
