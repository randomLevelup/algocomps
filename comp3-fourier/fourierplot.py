import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import asyncio
import threading

sample_rate = 44100
freq_base = 220.0
period = 0.1
num_harmonics = 64  # Number of harmonics to use in Fourier representation
displayed_harmonics = 64  # Number of harmonics to track
display_step = 4    # Show every other harmonic

def signal(type, t, freq):
    if type == 'sine':
        return np.sin(2 * np.pi * freq * t / sample_rate)
    elif type == 'saw':
        return 2 * (freq * ((t / sample_rate) % 0.01) - 0.5)
    elif type == 'tri':
        return abs(2 * (freq * ((t / sample_rate) % 0.01) - 0.5)) - 1
    else:
        raise ValueError("Unknown signal type")

def generate_chord_signals(freq_base: float, chords: dict[str, list[float]], period: float) -> dict:
    """
    Generate the raw signals for each chord and compute their Fourier coefficients
    """
    frame_length = int(sample_rate * period)
    chord_data = {}
    
    for name, intervals in chords.items():
        # Generate the time-domain signal for this chord
        buf = np.zeros(frame_length)
        for it in intervals:
            freq = freq_base * it
            t = np.arange(frame_length)
            buf += signal('sine', t, freq)
        
        # Normalize the signal
        if np.max(np.abs(buf)) > 0:
            buf = buf / np.max(np.abs(buf)) * 0.9
            
        # Compute the Fourier transform
        fft_result = np.fft.rfft(buf)
        
        # Store both the time domain signal and its FFT
        chord_data[name] = {
            'signal': buf,
            'fft': fft_result[:num_harmonics]  # Keep only the first num_harmonics
        }
        
    return chord_data

def get_frame(chord_data: dict, weights: np.ndarray) -> np.ndarray:
    """
    Create a frame by interpolating between the harmonic coefficients
    of the chord signals based on the weights
    """
    # Get the length of the FFT array from the first chord
    first_chord = list(chord_data.values())[0]
    fft_length = len(first_chord['fft'])
    frame_length = len(first_chord['signal'])
    
    # Interpolate between the Fourier coefficients
    interpolated_fft = np.zeros(fft_length, dtype=complex)
    
    for i, (name, data) in enumerate(chord_data.items()):
        interpolated_fft += data['fft'] * weights[i]
    
    # Store the current harmonics for visualization
    global current_harmonics
    current_harmonics = np.abs(interpolated_fft[:displayed_harmonics])
    
    # Inverse FFT to get the time domain signal
    frame = np.fft.irfft(interpolated_fft, n=frame_length)
    
    # Ensure the result is real and normalized
    frame = np.real(frame)
    if np.max(np.abs(frame)) > 0:
        frame = frame / np.max(np.abs(frame)) * 0.9
        
    return frame

# Update chord based on cursor position
def update_weights(cursor_pos):
    global current_weights, weights_changed
    
    distances = np.array([np.linalg.norm(cursor_pos - vertex) for vertex in vertices])
    
    # Calculate weights (closer = higher weight)
    weights = 1.0 / (distances + 0.001)  # Avoids division by zero
    weights = weights / np.sum(weights)  # Normalize
    
    # Only update if weights have changed significantly
    if np.allclose(weights, current_weights, rtol=0.01):
        return weights
    
    current_weights = weights.copy()
    weights_changed = True
    return weights

# Callback function for sounddevice
def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, buffer_lock
    
    with buffer_lock:
        if len(audio_buffer) < frames:
            print(f"Buffer underrun: need {frames}, have {len(audio_buffer)}")
            outdata.fill(0)
            return
        
        # Copy samples from buffer to output
        data = audio_buffer[:frames].copy()
        audio_buffer = audio_buffer[frames:]
    
    outdata[:] = data.reshape(-1, 1)

def update_cursor_position(event):
    global cursor_pos
    if event.inaxes == ax_main:  # Update to use ax_main instead of ax
        cursor_pos = np.array([event.xdata, event.ydata])
        # Don't update visual elements here - let animation handle it
        update_weights(cursor_pos)

# Async coroutines for audio and plot
async def plot_loop():
    global current_frame, weights_changed, current_harmonics
    
    while not stop_event.is_set():
        if weights_changed:
            new_frame = get_frame(chord_data, current_weights)
            weights_changed = False
            current_frame = new_frame
            
            # Just calculate the harmonics but don't update visual elements
            for i in range(len(displayed_indices)):
                harmonic_index = displayed_indices[i]
                if harmonic_index < len(current_harmonics):
                    # Store normalized values for the animation to use
                    normalized = np.clip(current_harmonics[harmonic_index] / max_amplitude, 0, 1)
                    harmonic_heights[i] = 0.8 + (normalized * 0.2)
        
        await asyncio.sleep(0.05)  # Plot updates every 50ms

async def audio_buffer_loop():
    global audio_buffer, current_frame
    
    buffer_target = int(sample_rate * 0.5)  # 500ms buffer capacity
    
    while not stop_event.is_set():
        with buffer_lock:
            if current_frame is None:
                await asyncio.sleep(0.01)
                continue
                
            # Add more audio if buffer is getting low
            if len(audio_buffer) < buffer_target:
                samples_needed = buffer_target - len(audio_buffer)
                frames_to_add = max(1, samples_needed // len(current_frame))
                
                for i in range(frames_to_add):
                    audio_buffer = np.concatenate((audio_buffer, current_frame))
        
        await asyncio.sleep(0.01)  # 10ms sleep

# Main setup
chords = {
    'Imaj7': [1/1, 5/4, 3/2, 15/8],
    'ii7': [9/8, 4/3, 5/3, 2/1],
    'V7': [3/2, 15/8, 9/8, 4/3],
    'VI7': [5/3, 25/12, 5/4, 3/2]
}

# Generate chord signals and compute their Fourier transforms
chord_data = generate_chord_signals(freq_base, chords, period)

# Initialize the current harmonics array
current_harmonics = np.zeros(displayed_harmonics)
# Array to store harmonic heights for animation
displayed_indices = range(0, displayed_harmonics, display_step)
harmonic_heights = np.zeros(len(displayed_indices))

# Find the maximum amplitude across all chords for normalization
max_amplitude = 0
for data in chord_data.values():
    max_amplitude = max(max_amplitude, np.max(np.abs(data['fft'][:displayed_harmonics])))

# Chord vertex layout
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
chord_names = list(chords.keys())
current_weights = np.ones(len(chord_names)) / len(chord_names)  # Start with equal weights

# Get initial frame
current_frame = get_frame(chord_data, current_weights)
audio_buffer = np.tile(current_frame, 3)  # Start with 3 copies

buffer_lock = threading.Lock()
weights_changed = True
stop_event = threading.Event()

# Create figure with two subplots: amplitude meter on top, chord plot below
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])

# Top subplot for amplitude meter
ax_meter = fig.add_subplot(gs[0])
# Only create bars for every other harmonic
displayed_indices = range(0, displayed_harmonics, display_step)
num_bars = len(displayed_indices)
x_positions = np.arange(num_bars)
amplitude_bars = ax_meter.bar(
    x_positions,
    np.zeros(num_bars),
    width=0.7,
    color='skyblue',
    edgecolor='navy'
)
# Set ylim to the more zoomed range
ax_meter.set_ylim(0.8, 1.0)
ax_meter.set_xlim(-0.5, num_bars-0.5)
ax_meter.set_title(f"Harmonic Amplitudes (Base Freq: {freq_base:.1f}Hz)")
ax_meter.set_xticks(x_positions)
ax_meter.set_xticklabels([f"{i+1}" for i in displayed_indices])
ax_meter.set_ylabel("Amplitude (scaled)")
ax_meter.set_xlabel("Harmonic Number")

# Bottom subplot for chord plot
ax_main = fig.add_subplot(gs[1])
scatter = ax_main.scatter(vertices[:, 0], vertices[:, 1], s=120, alpha=0.7) 

for i, name in enumerate(chord_names):
    ax_main.text(vertices[i, 0], vertices[i, 1] + 0.07, name, fontsize=12,
            ha='center', va='center')
    
ax_main.set_xticks([])
ax_main.set_yticks([])
ax_main.set_xlim(-0.2, 1.2)
ax_main.set_ylim(-0.2, 1.2)
ax_main.set_title('Move cursor to blend between chords using Fourier transforms')
cursor_point, = ax_main.plot([0.5], [0.5], 'ro', markersize=10)

# Track cursor position
cursor_pos = np.array([0.5, 0.5])
weights_text = ax_main.text(0.5, -0.1, "", transform=ax_main.transAxes, ha='center')

fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)
fig.canvas.mpl_connect('close_event', lambda event: stop_event.set())
fig.tight_layout(pad=3.0)

def animate(i):
    # Update cursor position
    cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
    
    # Update weights text
    weight_str = " ".join([f"{chord_names[i]}: {w:.2f}" for i, w in enumerate(current_weights)])
    weights_text.set_text(weight_str)
    
    # Update all amplitude bars
    for i, bar in enumerate(amplitude_bars):
        bar.set_height(harmonic_heights[i])
    
    # Return all animated elements
    return [cursor_point, weights_text, *amplitude_bars]

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)

# Coroutine event loop
def start_coroutines():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    tasks = [
        loop.create_task(plot_loop()),
        loop.create_task(audio_buffer_loop())
    ]
    
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()

# Start coroutines in a separate thread
coroutine_thread = threading.Thread(target=start_coroutines)
coroutine_thread.daemon = True
coroutine_thread.start()

# Start audio stream
stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024
)

with stream:
    plt.show()

# Cleanup
stop_event.set()
if coroutine_thread.is_alive():
    coroutine_thread.join(timeout=1.0)
