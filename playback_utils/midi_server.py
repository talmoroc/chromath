import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from mido import open_output, Message # type: ignore

app = FastAPI()

# --- MIDI Setup ---
# On macOS, virtual=True creates a new port. 
# On Windows, set virtual=False and use the name of your loopMIDI port.
try:
    output_port = open_output('Python MIDI Server', virtual=True)
    print("Virtual MIDI port created: Python MIDI Server")
except NotImplementedError:
    # Fallback for Windows (Assumes a loopMIDI port named 'Python MIDI')
    output_port = open_output('Python MIDI')
    print("Connected to existing MIDI port: Python MIDI")

# --- Data Models ---
class MidiNote(BaseModel):
    note: int      # 0-127
    velocity: int  # 0-127
    duration: float = 0.5 # seconds
    channel: int = 0      # 0-15

# --- Helper Functions ---
async def send_midi_note(data: MidiNote):
    """Sends Note On, waits, then sends Note Off asynchronously."""
    on_msg = Message('note_on', note=data.note, velocity=data.velocity, channel=data.channel)
    off_msg = Message('note_off', note=data.note, velocity=0, channel=data.channel)
    
    output_port.send(on_msg)
    await asyncio.sleep(data.duration)
    output_port.send(off_msg)

# --- API Endpoints ---
@app.post("/play-note")
async def play_note(note_data: MidiNote, background_tasks: BackgroundTasks):
    # We use BackgroundTasks so the API returns a response immediately 
    # while the note finishes playing in the background.
    background_tasks.add_task(send_midi_note, note_data)
    return {"status": "Message Sent", "note": note_data.note}

@app.post("/cc")
async def control_change(cc: int, value: int, channel: int = 0):
    """Send a Control Change message (e.g., for filters/knobs)"""
    msg = Message('control_change', control=cc, value=value, channel=channel)
    output_port.send(msg)
    return {"status": "CC Sent", "cc": cc, "value": value}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)