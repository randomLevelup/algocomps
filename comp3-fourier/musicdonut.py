import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import asyncio
import threading
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Circle
from PIL import Image
from music21 import *
import music21
import sys

sample_rate = 44100
grid_res = 512
num_harmonics = 64
circle_radius = 0.05
frame_duration = 0.15
frame_length = int(sample_rate * frame_duration)
image_path = 'C:\\Users\\jwest\\Desktop\\algocomps\\comp3-fourier\\images\\bumblebee.png'
freq_base = 110


def scale_amplitude(x, i, num_harmonics, center=0.55, bound=0.8, spread=5):
    norm = x * np.exp(-((i - (num_harmonics * center)) ** 2) / (2 * (spread ** 2)))
    if norm < 0.2:
        return x
    return x

# load and process image
try:
    img = Image.open(image_path).convert('L')
    img = img.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
    image_data = np.flip(np.array(img)) / 255.0
    image_data = image_data.T
    print(f"Image '{image_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'. Using a blank image.")
    image_data = np.zeros((grid_res, grid_res))
except Exception as e:
    print(f"Error loading image: {e}. Using a blank image.")
    image_data = np.zeros((grid_res, grid_res))

x_coords = np.linspace(0, 1, grid_res)
y_coords = np.linspace(0, 1, grid_res)

image_interpolator = RegularGridInterpolator((x_coords, y_coords), image_data,
                                           method='linear', bounds_error=False, fill_value=0)

def get_brightness_at_point(x, y):
    point = np.array([[np.clip(x, 0, 1), np.clip(y, 0, 1)]])
    return image_interpolator(point)[0]

def get_harmonics_from_image(center_x, center_y, radius, num_samples):
    amplitudes = np.zeros(num_samples)
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    for i, angle in enumerate(angles):
        sample_x = center_x + radius * np.cos(angle)
        sample_y = center_y + radius * np.sin(angle)
        brightness_val = -get_brightness_at_point(sample_x, sample_y) + 1
        amplitudes[i] = np.clip(brightness_val, 0, 1)
        amplitudes[i] = scale_amplitude(amplitudes[i], i, num_samples, center=0.6, bound=0, spread=15)
    return amplitudes

def generate_audio_frame(amplitudes, length):
    global current_audio_frame_for_plot
    num_amps = len(amplitudes)
    fft_coeffs = np.zeros(length // 2 + 1, dtype=complex)

    fft_coeffs[1 : num_amps + 1] = amplitudes

    frame = np.fft.irfft(fft_coeffs, n=length)

    max_amp = np.max(np.abs(frame))
    if max_amp > 0:
        frame /= max_amp

    current_audio_frame_for_plot = frame.copy()
    return frame * 10 # additional gain

audio_buffer = np.array([])
buffer_lock = threading.Lock()
current_harmonics = np.zeros(num_harmonics)
harmonics_changed = threading.Event()
cum_harmonics = []
current_audio_frame_for_plot = np.zeros(frame_length)
stop_event = threading.Event()

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

y_positions = np.arange(num_harmonics)
amplitude_bars = ax_amp.barh(y_positions, np.zeros(num_harmonics), height=0.7, color='skyblue', align='center')
ax_amp.set_xlim(0, 1.1)
ax_amp.set_ylim(-0.5, num_harmonics - 0.5)
ax_amp.set_title("Amplitudes")
ax_amp.set_yticks(y_positions[::4])
ax_amp.set_yticklabels([f"{i+1}" for i in y_positions[::4]])
ax_amp.set_xlabel("Amplitude")
ax_amp.invert_yaxis()

im = ax_main.imshow(image_data.T, origin='lower', extent=(0, 1, 0, 1), cmap='gray', alpha=1.0)
ax_main.set_title('Image -> Harmonics')
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

signal_time = np.linspace(0, frame_duration, frame_length)
signal_line, = ax_signal.plot(signal_time, current_audio_frame_for_plot, lw=1)
ax_signal.set_ylim(-1.1, 1.1)
ax_signal.set_xlim(0, frame_duration)
ax_signal.set_title("Output Signal Frame")
ax_signal.set_xlabel("Time (s)")
ax_signal.set_ylabel("Amplitude")
ax_signal.grid(True, linestyle=':', alpha=0.6)

current_harmonics = get_harmonics_from_image(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)
generate_audio_frame(current_harmonics, frame_length)

def update_cursor_position(event):
    global cursor_pos, current_harmonics, cum_harmonics
    if event.inaxes == ax_main:
        new_pos = np.array([event.xdata, event.ydata])
        if not np.allclose(new_pos, cursor_pos, atol=0.005):
            cursor_pos = new_pos
            current_harmonics = get_harmonics_from_image(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)
            cum_harmonics.append(current_harmonics)
            harmonics_changed.set()

def animate(i):
    cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
    sampling_circle.center = cursor_pos[0], cursor_pos[1]

    for bar, h in zip(amplitude_bars, current_harmonics):
        bar.set_width(h)

    signal_line.set_ydata(current_audio_frame_for_plot)

    return [cursor_point, sampling_circle, signal_line, *amplitude_bars]

ani = animation.FuncAnimation(fig, animate, interval=30, blit=True)

async def audio_buffer_loop():
    global audio_buffer, current_harmonics
    buffer_target = int(sample_rate * 0.2)

    while not stop_event.is_set():
        if harmonics_changed.is_set():
            current_audio_frame = generate_audio_frame(current_harmonics, frame_length)
            harmonics_changed.clear()
        elif 'current_audio_frame' not in locals():
            current_audio_frame = generate_audio_frame(current_harmonics, frame_length)

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

def start_coroutines():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = [
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

coroutine_thread = threading.Thread(target=start_coroutines)
coroutine_thread.daemon = True
coroutine_thread.start()

fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)
fig.canvas.mpl_connect('close_event', lambda event: stop_event.set())
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

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



def frameToChord(frame, notes, division):
    #set list of pitches to most prominent frequencies
    highestAmplitudes = [(i, frame[i]) for i in range(len(frame))]
    highestAmplitudes.sort(key = (lambda x : x[1]), reverse=True)
    pitches = []
    for i in range(notes):
        print(highestAmplitudes[i][1])
        print((highestAmplitudes[i][0] + 1) * freq_base)
        tone = pitch.Pitch()
        tone.frequency = (highestAmplitudes[i][0] + 1) * freq_base
        tone = tone.transpose(-48)
        pitches.append(tone)



    c = chord.Chord(pitches)
    d = duration.Duration(1/3)
    d = duration.Duration(4/division)
    c.duration = d
    return c

def main():


    print("Welcome to the Donut Jungle!")
    print("Before we can listen to the song you just created you need to fill in a few parameters")
    division = int(input("Please enter what division you would like each melody chord to be (e.x eighth-note = 8, quarter = 4, triplets = 12) - this will effect the overall length of your piece "))
    notesInChord = int(input("Please enter how many notes you would like to capture from each frame (e.x 4) "))

    finalStream = music21.stream.Stream()
    finalStream.insert(0, instrument.ElectricPiano())
    global cum_harmonics
    frames = cum_harmonics
    for frame in frames:
        finalStream.append(frameToChord(frame, notesInChord, division))

    if ("-m" in sys.argv):
        finalStream.show("midi")
    elif ("-s" in sys.argv):
        finalStream.show()
    else:
        finalStream.show("text")


if __name__ == "__main__":
    main()
