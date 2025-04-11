import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.widgets import Button
import matplotlib.animation as animation

sample_rate = 44100
duration = 1.0
time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
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

def audio_callback(outdata, frames, _time_info, _status_flags):
    global callback_idx
    chunk = current_chord[callback_idx:callback_idx+frames]
    if len(chunk) < frames:
        remainder = frames - len(chunk)
        chunk = np.concatenate((chunk, current_chord[:remainder]))
        callback_idx = remainder
    else:
        callback_idx += frames
        callback_idx %= len(current_chord)
    outdata[:] = chunk.reshape(-1, 1)

freq_base = 220

chords = {
    'Imaj7': [1, 5/4, 3/2, 15/8],
    'ii7': [9/8, 4/3, 5/3, 2],
    'VDom7': [3/2, 15/8, 9/8, 4/3],
    'VIDom7': [5/3, 25/12, 5/4, 3/2]
}

# Set up vertices for the chords in a square formation
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
chord_names = list(chords.keys())
chord_weights = np.zeros(len(chord_names))
current_chord = np.zeros_like(time)

# Function to update chord based on cursor position
def update_chord(cursor_pos):
    global current_chord
    # Calculate distances to each vertex
    distances = np.array([np.linalg.norm(cursor_pos - vertex) for vertex in vertices])
    
    # Convert distances to weights (closer = higher weight)
    weights = 1.0 / (distances + 0.1)  # Adding 0.1 to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights
    
    # Generate combined chord with smooth transition
    new_chord = np.zeros_like(time)
    for i, name in enumerate(chord_names):
        new_chord += weights[i] * generate_chord(freq_base, chords[name])
    
    # Normalize the new chord
    new_chord = new_chord / np.max(np.abs(new_chord))
    
    # Smoothly transition to avoid clicks (simple crossfade)
    crossfade_samples = int(sample_rate * 0.05)  # 50ms crossfade
    if crossfade_samples > 0:
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)
        
        # Apply crossfade to beginning of the buffer
        new_chord[:crossfade_samples] = (new_chord[:crossfade_samples] * fade_in + 
                                        current_chord[:crossfade_samples] * fade_out)
    
    current_chord = new_chord
    return weights

# Initialize with first chord
current_chord = generate_chord(freq_base, chords[chord_names[0]])

# Set up the interactive plot
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(vertices[:, 0], vertices[:, 1], s=200, alpha=0.7)

# Add labels for each chord
for i, name in enumerate(chord_names):
    ax.text(vertices[i, 0], vertices[i, 1], name, fontsize=12, 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

# Set plot limits and title
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_title('Move cursor to blend between chords')
cursor_point, = ax.plot([0.5], [0.5], 'ro', markersize=10)  # Initialize with arrays

# Variables to track cursor position
cursor_pos = np.array([0.5, 0.5])
weights_text = ax.text(0.5, -0.1, "", transform=ax.transAxes, ha='center')

def update_cursor_position(event):
    global cursor_pos
    if event.inaxes == ax:
        cursor_pos = np.array([event.xdata, event.ydata])
        cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])  # Pass arrays instead of scalars
        
        # Update chord weights based on new cursor position
        weights = update_chord(cursor_pos)
        
        # Update the weights text
        weight_str = " ".join([f"{chord_names[i]}: {w:.2f}" for i, w in enumerate(weights)])
        weights_text.set_text(weight_str)
        
        fig.canvas.draw_idle()

# Connect the event handler
fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)

# Animation function to keep the plot responsive
def animate(i):
    return cursor_point, weights_text

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)

# Start audio stream
with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
    plt.show()

