import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.animation as animation
from collections import deque

sample_rate = 44100
duration = 0.1
time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
buffer_queue = deque(maxlen=10)

callback_idx = 0

def signal(f):
    res = np.zeros_like(time)
    for t in range(len(time)):
        res[t] = np.sin(2 * np.pi * f * time[t])
    return res

def mux(signals):
    res = np.zeros_like(time)
    for t in range(len(time)):
        for sig in signals:
            res[t] += sig[t]
    return res / np.max(np.abs(res))

def generate_chord(f, intervals):
    pitches = []
    for it in intervals:
        pitches.append(signal(f * it))
    return mux(pitches)

def find_zero_crossings(audio_data):
    return np.where(np.diff(np.signbit(audio_data)))[0]

def audio_callback(outdata, frames, time_info, status_flags):
    if len(buffer_queue) == 0:
        # If queue is empty, generate silence
        outdata[:] = np.zeros((frames, 1))
        return
    
    current_buffer = buffer_queue.popleft()
    
    # If buffer is smaller than required frames, pad with zeros
    if len(current_buffer) < frames:
        padding = np.zeros(frames - len(current_buffer))
        current_buffer = np.concatenate((current_buffer, padding))
    
    # If buffer is larger than required frames, take only what we need
    elif len(current_buffer) > frames:
        leftover = current_buffer[frames:]
        current_buffer = current_buffer[:frames]
        buffer_queue.appendleft(leftover)
        
    outdata[:] = current_buffer.reshape(-1, 1)

freq_base = 220

chords = {
    'Imaj7': [1, 5/4, 3/2, 15/8],
    'ii7': [9/8, 4/3, 5/3, 2],
    'VDom7': [3/2, 15/8, 9/8, 4/3],
    'VIDom7': [5/3, 25/12, 5/4, 3/2]
}

# chord vertex layout
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
chord_names = list(chords.keys())
current_weights = np.ones(len(chord_names)) / len(chord_names)  # Initialize with equal weights
last_chord = None

# update chord based on cursor position
def update_chord(cursor_pos):
    global current_weights, last_chord
    
    distances = np.array([np.linalg.norm(cursor_pos - vertex) for vertex in vertices])
    
    # calculate weights (closer = higher weight)
    weights = 1.0 / (distances + 0.1)  # avoids division by zero
    weights = weights / np.sum(weights)  # normalize
    
    # only update if weights have changed significantly
    if last_chord is not None and np.allclose(weights, current_weights, rtol=0.01):
        return weights
    
    current_weights = weights
    
    # generate chord segment
    new_chord = np.zeros_like(time)
    for i, name in enumerate(chord_names):
        new_chord += weights[i] * generate_chord(freq_base, chords[name])
    
    new_chord = new_chord / np.max(np.abs(new_chord) + 0.000001) # normalize
    
    # find zero crossing for smooth transition
    if last_chord is not None:
        crossings = find_zero_crossings(last_chord[-100:])
        cross_idx = crossings[-1]
        
        # Keep the part before the zero crossing from the last buffer
        keep_last = last_chord[:-100 + cross_idx + 1]
        buffer_queue.append(keep_last)
        
        buffer_queue.append(new_chord)
    else:
        # initial buffer
        buffer_queue.append(new_chord)
    
    last_chord = new_chord
    return weights

# initialize buffer
initial_chord = generate_chord(freq_base, chords[chord_names[0]])
buffer_queue.append(initial_chord)
last_chord = initial_chord

# setup interactive plot
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(vertices[:, 0], vertices[:, 1], s=120, alpha=0.7) 

for i, name in enumerate(chord_names):
    ax.text(vertices[i, 0], vertices[i, 1] + 0.07, name, fontsize=12,
            ha='center', va='center')
    
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_title('Move cursor to blend between chords')
cursor_point, = ax.plot([0.5], [0.5], 'ro', markersize=10)

# track cursor position
cursor_pos = np.array([0.5, 0.5])
weights_text = ax.text(0.5, -0.1, "", transform=ax.transAxes, ha='center')

def update_cursor_position(event):
    global cursor_pos
    if event.inaxes == ax:
        cursor_pos = np.array([event.xdata, event.ydata])
        cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
        
        weights = update_chord(cursor_pos)
        
        weight_str = " ".join([f"{chord_names[i]}: {w:.2f}" for i, w in enumerate(weights)])
        weights_text.set_text(weight_str)
        
        fig.canvas.draw_idle()

# canvas event handler
fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)

# animate plot
def animate(i):
    return cursor_point, weights_text

ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)

# periodically update the buffer even when cursor is not moving
def maintain_buffer(event):
    update_chord(cursor_pos)

# maintain_buffer every 50ms to keep the buffer filled
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(maintain_buffer, None)
timer.start()

# start audio stream
with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, 
                     blocksize=1024):
    plt.show()

