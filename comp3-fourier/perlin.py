import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import asyncio
import threading
import math
import random
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Circle

sample_rate = 44100
perlin_res = 512 # resolution of the noise grid
perlin_scale = 20
num_harmonics = 64 # number of harmonics (and sample points)
circle_radius = 0.15 # radius of the sampling circle in plot coordinates
frame_duration = 0.1 # desired frame duration in seconds
frame_length = int(sample_rate * frame_duration) # samples per audio frame generation

# --- perlin noise implementation ---
def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, x):
    return a + x * (b - a)

def gradient(h, x, y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]])
    g = vectors[h % 8]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    xi, yi = x.astype(int), y.astype(int)
    xf, yf = x - xi, y - yi

    u = fade(xf)
    v = fade(yf)

    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)

    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

def scale_amplitude(x, i, num_harmonics, center=0.55, bound=0.8, spread=5):
    if i <= 1:
        return 0.1
    norm = x * np.exp(-((i - (num_harmonics * center)) ** 2) / (2 * (spread ** 2)))
    if norm < 0.1:
        return 0.1
    return (1 - bound) * (norm - 1) + 1

# generate noise field
lin = np.linspace(0, perlin_scale, perlin_res, endpoint=False)
x_coords = np.linspace(0, 1, perlin_res) # use 0-1 coordinates for interpolation
y_coords = np.linspace(0, 1, perlin_res)
x_grid, y_grid = np.meshgrid(lin, lin) # grid for noise generation
noise_field = perlin(x_grid, y_grid, seed=random.randint(0, 100))

# create interpolation function for the noise field
# use bounds_error=false and fill_value=0
noise_interpolator = RegularGridInterpolator((x_coords, y_coords), noise_field.T,
                                             method='linear', bounds_error=False, fill_value=0)

# function to get noise value at a specific point using interpolation
def get_noise_at_point(x, y):
    point = np.array([[np.clip(x, 0, 1), np.clip(y, 0, 1)]])
    return noise_interpolator(point)[0]

# function to sample noise around the circle and get harmonic amplitudes
def get_harmonics_from_noise(center_x, center_y, radius, num_samples):
    amplitudes = np.zeros(num_samples)
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    for i, angle in enumerate(angles):
        sample_x = center_x + radius * np.cos(angle)
        sample_y = center_y + radius * np.sin(angle)
        noise_val = get_noise_at_point(sample_x, sample_y)
        # map noise value (approx -0.7 to 0.7) to amplitude (0 to 1)
        amplitudes[i] = np.clip((noise_val + 0.7) / 1.4, 0, 1)
        # scale for better harmonic effect
        amplitudes[i] = scale_amplitude(amplitudes[i], i, num_samples, center=0.65, bound=0.7, spread=6.5)
    return amplitudes

# function to generate an audio frame from harmonic amplitudes using ifft
def generate_audio_frame(amplitudes, length):
    global current_audio_frame_for_plot
    num_amps = len(amplitudes)
    fft_coeffs = np.zeros(length // 2 + 1, dtype=complex)

    # assign amplitudes to the magnitudes of the first num_amps harmonics
    fft_coeffs[1 : num_amps + 1] = amplitudes

    frame = np.fft.irfft(fft_coeffs, n=length)

    max_amp = np.max(np.abs(frame))
    if max_amp > 0:
        frame /= max_amp

    current_audio_frame_for_plot = frame.copy()
    return frame * 10 # gain

# audio buffer setup
audio_buffer = np.array([])
buffer_lock = threading.Lock()
current_harmonics = np.zeros(num_harmonics)
harmonics_changed = threading.Event()
current_audio_frame_for_plot = np.zeros(frame_length)
stop_event = threading.Event()

# audio callback function
def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, buffer_lock
    with buffer_lock:
        if len(audio_buffer) < frames:
            outdata[:] = np.zeros((frames, 1))
            audio_buffer = np.array([])
            return

        data = audio_buffer[:frames].copy()
        audio_buffer = audio_buffer[frames:]

    outdata[:] = data.reshape(-1, 1)


fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], wspace=0.3, hspace=0.3)

ax_main = fig.add_subplot(gs[0, 0])
ax_amp = fig.add_subplot(gs[0, 1])
ax_signal = fig.add_subplot(gs[1, 0])

# amplitude plot setup (vertical)
y_positions = np.arange(num_harmonics)
amplitude_bars = ax_amp.barh(y_positions, np.zeros(num_harmonics), height=0.7, color='skyblue', align='center')
ax_amp.set_xlim(0, 1.1)
ax_amp.set_ylim(-0.5, num_harmonics - 0.5)
ax_amp.set_title("Amplitudes")
ax_amp.set_yticks(y_positions[::4])
ax_amp.set_yticklabels([f"{i+1}" for i in y_positions[::4]])
ax_amp.set_xlabel("Amplitude")
ax_amp.invert_yaxis()

# main plot setup
im = ax_main.imshow(noise_field, origin='lower', extent=(0, 1, 0, 1), cmap='viridis', alpha=0.8)
ax_main.set_title('Noise -> Harmonics')
ax_main.set_xlabel('X Coordinate')
ax_main.set_ylabel('Y Coordinate')
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 1)
ax_main.set_aspect('equal', adjustable='box')

cursor_point, = ax_main.plot([0.5], [0.5], 'ro', markersize=8)
cursor_pos = np.array([0.5, 0.5])
sampling_circle = Circle((cursor_pos[0], cursor_pos[1]), circle_radius,
                         edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5)
ax_main.add_patch(sampling_circle)

# signal plot setup
signal_time = np.linspace(0, frame_duration, frame_length)
signal_line, = ax_signal.plot(signal_time, current_audio_frame_for_plot, lw=1)
ax_signal.set_ylim(-1.1, 1.1)
ax_signal.set_xlim(0, frame_duration)
ax_signal.set_title("Output Signal Frame")
ax_signal.set_xlabel("Time (s)")
ax_signal.set_ylabel("Amplitude")
ax_signal.grid(True, linestyle=':', alpha=0.6)


current_harmonics = get_harmonics_from_noise(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)
generate_audio_frame(current_harmonics, frame_length)
harmonics_changed.set()

# update cursor position based on mouse event
def update_cursor_position(event):
    global cursor_pos, current_harmonics
    if event.inaxes == ax_main:
        new_pos = np.array([event.xdata, event.ydata])
        if not np.allclose(new_pos, cursor_pos, atol=0.005):
             cursor_pos = new_pos
             current_harmonics = get_harmonics_from_noise(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)
             harmonics_changed.set()

def animate(i):
    cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
    sampling_circle.center = cursor_pos[0], cursor_pos[1]

    for bar, h in zip(amplitude_bars, current_harmonics):
        bar.set_width(h)

    signal_line.set_ydata(current_audio_frame_for_plot)

    return [cursor_point, sampling_circle, signal_line, *amplitude_bars]

ani = animation.FuncAnimation(fig, animate, interval=30, blit=True)

# async coroutines
async def plot_loop():
    while not stop_event.is_set():
        await asyncio.sleep(0.1)

async def audio_buffer_loop():
    global audio_buffer
    buffer_target = int(sample_rate * 0.2)
    current_audio_frame = generate_audio_frame(current_harmonics, frame_length)

    while not stop_event.is_set():
        if harmonics_changed.is_set():
            current_audio_frame = generate_audio_frame(current_harmonics, frame_length)
            harmonics_changed.clear()

        with buffer_lock:
            if len(audio_buffer) < buffer_target:
                 if len(current_audio_frame) > 0:
                     samples_needed = buffer_target - len(audio_buffer)
                     frames_to_add = max(1, samples_needed // len(current_audio_frame))
                     for _ in range(frames_to_add):
                         audio_buffer = np.concatenate((audio_buffer, current_audio_frame))
                 else:
                     await asyncio.sleep(0.005)

        await asyncio.sleep(0.01)

# coroutine event loop setup
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
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        print("Coroutine loop closed.")

# start coroutines in a separate thread
coroutine_thread = threading.Thread(target=start_coroutines)
coroutine_thread.daemon = True
coroutine_thread.start()

# connect mouse motion and close events
fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)
fig.canvas.mpl_connect('close_event', lambda event: stop_event.set())
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


# start audio stream
try:
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback,
        blocksize=1024
    )
    with stream:
        print("Audio stream started. Close the plot window to stop.")
        plt.show()

except Exception as e:
    print(f"Error starting audio stream: {e}")
    plt.show()


# cleanup
print("Plot window closed, stopping...")
stop_event.set()
print("Waiting for coroutine thread to finish...")
if coroutine_thread.is_alive():
    coroutine_thread.join(timeout=2.0)
    if coroutine_thread.is_alive():
        print("Coroutine thread did not finish cleanly.")
    else:
        print("Coroutine thread finished.")
print("Cleanup complete. Exiting.")
