import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import asyncio
import threading

sample_rate = 44100
freq_base = 220.0
period = 0.1

def get_frame(freq_base: float, chords: dict[str, list[float]], weights: np.ndarray, period: float) -> np.ndarray:
    signals = []
    frame_length = int(sample_rate * period)
    for i, (name, intervals) in enumerate(chords.items()):
        buf = np.zeros(frame_length)
        for it in intervals:
            freq = freq_base * it
            t = np.arange(frame_length)
            buf += np.sin(2 * np.pi * freq * t / sample_rate)
        signals.append(buf * weights[i])
    
    res = np.zeros_like(signals[0])
    for sig in signals:
        res += sig
    
    # normalize
    if np.max(np.abs(res)) > 0:
        res = res / np.max(np.abs(res)) * 0.9  # leave some headroom
    
    # fade between frames
    fade_size = min(100, frame_length // 20)
    fade_in = np.linspace(0.0, 1.0, fade_size)
    fade_out = np.linspace(1.0, 0.0, fade_size)
    
    res[:fade_size] *= fade_in
    res[-fade_size:] *= fade_out
    
    return res

# update chord based on cursor position
def update_weights(cursor_pos):
    global current_weights, weights_changed
    
    distances = np.array([np.linalg.norm(cursor_pos - vertex) for vertex in vertices])
    
    # calculate weights (closer = higher weight)
    weights = 1.0 / (distances + 0.001)  # avoids division by zero
    weights = weights / np.sum(weights)  # normalize
    
    # only update if weights have changed significantly
    if np.allclose(weights, current_weights, rtol=0.01):
        return weights
    
    current_weights = weights.copy()
    weights_changed = True
    return weights

# callback function for sounddevice
def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, buffer_lock
    
    with buffer_lock:
        if len(audio_buffer) < frames:
            print(f"Buffer underrun: need {frames}, have {len(audio_buffer)}")
            outdata.fill(0)
            return
        
        # copy samples from buffer to output
        data = audio_buffer[:frames].copy()
        audio_buffer = audio_buffer[frames:]
    
    outdata[:] = data.reshape(-1, 1)

def update_cursor_position(event):
    global cursor_pos, weights_changed
    if event.inaxes == ax:
        cursor_pos = np.array([event.xdata, event.ydata])
        cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
        
        weights = update_weights(cursor_pos)
        weight_str = " ".join([f"{chord_names[i]}: {w:.2f}" for i, w in enumerate(weights)])
        weights_text.set_text(weight_str)
        
        fig.canvas.draw_idle()

# async coroutines for audio and plot
async def plot_loop():
    global current_frame, weights_changed
    
    while not stop_event.is_set():
        if weights_changed:
            new_frame = get_frame(freq_base, chords, current_weights, period)
            weights_changed = False
            current_frame = new_frame
        
        await asyncio.sleep(0.05) # plot updates every 50ms

async def audio_buffer_loop():
    global audio_buffer, current_frame, current_position
    
    buffer_target = int(sample_rate * 0.5)  # 500ms buffer capacity
    
    previous_frame_end = None
    crossfade_length = int(sample_rate * 0.15)  # 150ms crossfade
    
    while not stop_event.is_set():
        with buffer_lock:
            if current_frame is None:
                await asyncio.sleep(0.01)
                continue
                
            # add more audio if buffer is getting low
            if len(audio_buffer) < buffer_target:
                samples_needed = buffer_target - len(audio_buffer)
                frames_to_add = max(1, samples_needed // len(current_frame))
                
                for i in range(frames_to_add):
                    frame_to_add = current_frame
                    
                    # apply crossfade
                    if previous_frame_end is not None:
                        fade_length = min(crossfade_length, len(previous_frame_end), len(frame_to_add))
                        fade_in = np.linspace(0.0, 1.0, fade_length)
                        fade_out = np.linspace(1.0, 0.0, fade_length)
                        
                        frame_to_add[:fade_length] = (
                            frame_to_add[:fade_length] * fade_in + 
                            previous_frame_end[-fade_length:] * fade_out
                        )
                    
                    audio_buffer = np.concatenate((audio_buffer, frame_to_add))
                    previous_frame_end = frame_to_add[-crossfade_length*2:]
        
        await asyncio.sleep(0.01)  # 10ms sleep

# main setup
chords = {
    'Imaj7': [1/1, 5/4, 3/2, 15/8],
    'ii7': [9/8, 4/3, 5/3, 2/1],
    'V7': [3/2, 15/8, 9/8, 4/3],
    'VI7': [5/3, 25/12, 5/4, 3/2]
}

# chord vertex layout
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
chord_names = list(chords.keys())
current_weights = np.ones(len(chord_names)) / len(chord_names)  # start with equal weights

# get initial frame
current_frame = get_frame(freq_base, chords, current_weights, period)
audio_buffer = np.tile(current_frame, 3)  # start with 3 copies

buffer_lock = threading.Lock()
current_position = 0
weights_changed = True
stop_event = threading.Event()

# plot setup
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

fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)
fig.canvas.mpl_connect('close_event', lambda event: stop_event.set())

def animate(i):
    return cursor_point, weights_text

ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)

# coroutine event loop
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

# start coroutines in a separate thread
coroutine_thread = threading.Thread(target=start_coroutines)
coroutine_thread.daemon = True
coroutine_thread.start()

# start audio stream
stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024
)

with stream:
    plt.show()

# cleanup
stop_event.set()
if coroutine_thread.is_alive():
    coroutine_thread.join(timeout=1.0)
