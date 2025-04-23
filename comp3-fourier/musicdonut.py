"""
program.py

Authors: Jake Kerrigan, Jupiter Westbard, Milo Goldstein and Jason Miller

Program: Music From Noise

Brief: Given an image, this program opens an interactive plot where the user
       can 'play' the image by moving their cursor around. The brightnesses of
       pixels near the cursor are used to compute the most dominant harmonics
       that the user hears. Once the plot is closed, a MIDI/MusicXML file
       is generated based on the harmonics that were played.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
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
circle_radius = 0.1
frame_duration = 0.15
frame_length = int(sample_rate * frame_duration)
image_path = None
freq_base = 110

# load and process image
if len(sys.argv) < 2:
    print("Usage: python musicdonut.py <image_path> [-m | -s]")
    sys.exit(1)
elif sys.argv[1] in ["-m", "-s"]:
        print("Usage: python musicdonut.py <image_path> [-m | -s]")
        sys.exit(1)
else:
    image_path = sys.argv[1]

try:
    img = Image.open(image_path).convert('L') # convert to grayscale
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
    """ Get brightness value at a point in the image. """
    point = np.array([[np.clip(x, 0, 1), np.clip(y, 0, 1)]])
    return image_interpolator(point)[0]

def get_harmonics_from_image(center_x, center_y, radius, num_samples):
    """ Calculate harmonic amplitudes based on samples from the image """
    amplitudes = np.zeros(num_samples)
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    for i, angle in enumerate(angles):
        sample_x = center_x + radius * np.cos(angle)
        sample_y = center_y + radius * np.sin(angle)
        brightness_val = -get_brightness_at_point(sample_x, sample_y) + 1
        amplitudes[i] = np.clip(brightness_val, 0, 1)
    return amplitudes

def generate_audio_frame(amplitudes, length):
    """ Generate a 'frame' of audio by stacking harmonics with IFFT """
    global current_audio_frame_for_plot
    num_amps = len(amplitudes)
    fft_coeffs = np.zeros(length // 2 + 1, dtype=float)
    # use amplitudes as FFT coefficients
    fft_coeffs[1 : num_amps + 1] = amplitudes
    # do inverse FFT to get output signal
    # the fundamental is defined by the 'wavelength' a.k.a. the length of the frame
    frame = np.fft.irfft(fft_coeffs, n=length)
    max_amp = np.max(np.abs(frame))
    if max_amp > 0:
        frame /= max_amp
    current_audio_frame_for_plot = frame.copy()
    return frame * 10 # additional gain

audio_buffer = np.array([])
buffer_lock = threading.Lock()
current_harmonics = np.zeros(num_harmonics)
cum_harmonics = []

def audio_callback(outdata, frames, time_info, status):
    """ Callback function for live audio stream """
    global audio_buffer, buffer_lock, current_harmonics, cum_harmonics
    with buffer_lock:
        # generate more audio if buffer is running low
        while len(audio_buffer) < frames:
            new_frame = generate_audio_frame(current_harmonics, frame_length)
            if len(new_frame) > 0:
                audio_buffer = np.concatenate((audio_buffer, new_frame))
            else:
                break
        # if buffer is still too small after generation, output silence
        if len(audio_buffer) < frames:
            outdata[:] = np.zeros((frames, 1))
            audio_buffer = np.array([])
            return
        data_to_output = audio_buffer[:frames].copy()
        audio_buffer = audio_buffer[frames:]

    outdata[:] = data_to_output.reshape(-1, 1)
    # [DONT] update harmonics every frame
    # cum_harmonics.append(current_harmonics)


""" Matplot setup """

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], wspace=0.3, hspace=0.3)

ax_main = fig.add_subplot(gs[0, 0])
ax_amp = fig.add_subplot(gs[0, 1])
ax_signal = fig.add_subplot(gs[1, 0])

# set up amplitude bars
y_positions = np.arange(num_harmonics)
amplitude_bars = ax_amp.barh(y_positions, np.zeros(num_harmonics), height=0.7, color='skyblue', align='center')
ax_amp.set_xlim(0, 1.1)
ax_amp.set_ylim(-0.5, num_harmonics - 0.5)
ax_amp.set_title("Amplitudes")
ax_amp.set_yticks(y_positions[::4])
ax_amp.set_yticklabels([f"{i+1}" for i in y_positions[::4]])
ax_amp.set_xlabel("Amplitude")
ax_amp.invert_yaxis()

# set up main image plot
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

# set up signal plot
signal_time = np.linspace(0, frame_duration, frame_length)
signal_line, = ax_signal.plot(signal_time, np.zeros(frame_length), lw=1) # start signal plot with zeros
ax_signal.set_ylim(-1.1, 1.1)
ax_signal.set_xlim(0, frame_duration)
ax_signal.set_title("Output Signal Frame")
ax_signal.set_xlabel("Time (s)")
ax_signal.set_ylabel("Amplitude")
ax_signal.grid(True, linestyle=':', alpha=0.6)

current_harmonics = get_harmonics_from_image(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)

def update_cursor_position(event):
    """ Recalculate amplitudes cursor moves significantly """
    global cursor_pos, current_harmonics, cum_harmonics
    if event.inaxes == ax_main:
        new_pos = np.array([event.xdata, event.ydata])
        if not np.allclose(new_pos, cursor_pos, atol=0.005):
            cursor_pos = new_pos
            current_harmonics = get_harmonics_from_image(cursor_pos[0], cursor_pos[1], circle_radius, num_harmonics)
            # "record" current harmonics for later music generation
            cum_harmonics.append(current_harmonics)

def animate(i):
    """ Animation callback for interactive plot """
    plot_frame = generate_audio_frame(current_harmonics, frame_length)
    cursor_point.set_data([cursor_pos[0]], [cursor_pos[1]])
    sampling_circle.center = cursor_pos[0], cursor_pos[1]
    # update amplitude bars
    for bar, h in zip(amplitude_bars, current_harmonics):
        bar.set_width(h)
    # update signal line
    signal_line.set_ydata(plot_frame)
    return [cursor_point, sampling_circle, signal_line, *amplitude_bars]

ani = animation.FuncAnimation(fig, animate, interval=30, blit=True)

fig.canvas.mpl_connect('motion_notify_event', update_cursor_position)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# pre-fill the buffer
initial_frame = generate_audio_frame(current_harmonics, frame_length)
if len(initial_frame) > 0:
    with buffer_lock:
        # Add a few frames to start
        for _ in range(5):
             audio_buffer = np.concatenate((audio_buffer, initial_frame))

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

print("Plot window closed, stopping...")
print("Cleanup complete. Exiting.")

def frameToChord(frame, notes, division):
    """ Set list of pitches to most prominent frequencies """
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
    print("Welcome to the Harmonic Jungle!")
    print("Before we can listen to the song you just created you need to fill in a few parameters")
    division = int(input("Please enter what division you would like each melody chord to be (e.x eighth-note = 8, quarter = 4, triplets = 12) - this will affect the overall length of your piece "))
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
