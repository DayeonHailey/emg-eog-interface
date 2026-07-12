"""
eog_emg_visualizer.py

Real-time visualization and control pipeline for a dual-channel
EMG/EOG biosignal system streamed over UART from an MSP430.

Features:
- Live plotting of EMG and EOG signals
- EMG frequency-domain analysis (FFT) for tremor-band detection
- Two-phase EOG calibration (center baseline + left/right sampling)
- Direction classification (LEFT / CENTER / RIGHT) with persistence
  and opposite-direction blocking to reduce false transitions
- Blink detection via biphasic peak analysis on the EOG channel
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.patches import Circle, FancyBboxPatch
from collections import deque
import time
from scipy.fft import fft, fftfreq


# ===== Configuration =====
SERIAL_PORT = '/dev/tty.usbserial-XXXXXXXX'  # TODO: update to your device's serial port
BAUD_RATE = 115200
PACKET_SIZE = 9
HEADER = [0xAA, 0x77, 0xAA]

DISPLAY_SAMPLES = 2000
SAMPLING_RATE = 1000
UPDATE_INTERVAL = 50  # 20 FPS

DEBUG_MODE = False

SMART_Y_RANGES = {
    'emg': {'min': 500, 'max': 4095},
    'eog1': {'min': 500, 'max': 4095},
    'eog2': {'min': 500, 'max': 4095}
}

Y_AXIS_SMOOTHING = 0.1

# Direction detection persistence settings
DIRECTION_PERSISTENCE_SECONDS = 5  # Hold a detected direction for this long
DIRECTION_PERSISTENCE_FRAMES = int(DIRECTION_PERSISTENCE_SECONDS * (1000 / UPDATE_INTERVAL))

# Blink persistence settings
BLINK_PERSISTENCE_SECONDS = 2
BLINK_PERSISTENCE_FRAMES = int(BLINK_PERSISTENCE_SECONDS * (1000 / UPDATE_INTERVAL))

BLINK_THRESHOLD = 0.5

# Center stabilization settings
CENTER_STABILITY_THRESHOLD = 0.15   # Center dead-zone (+/- 15%)
TRANSITION_IGNORE_FRAMES = 10       # Frames to ignore right after a direction transition
OPPOSITE_DIRECTION_BLOCK_FRAMES = 30  # Frames to block the opposite direction (~1.5s)

# Data buffers
emg_buffer = deque(maxlen=DISPLAY_SAMPLES)
eog1_buffer = deque(maxlen=DISPLAY_SAMPLES)
eog2_buffer = deque(maxlen=DISPLAY_SAMPLES)
eog1_normalized_buffer = deque(maxlen=DISPLAY_SAMPLES)
eog2_normalized_buffer = deque(maxlen=DISPLAY_SAMPLES)

emg_fft_buffer = deque(maxlen=1000)

sample_count = 0
dominant_freq = 0.0

# Label tracking state
current_direction_label = "CENTER"
last_detected_direction = None
direction_timer = 0
transition_ignore_counter = 0
opposite_block_counter = 0    # Frames remaining for opposite-direction block
blocked_direction = None      # Direction currently being blocked

current_blink_label = "NO_BLINK"
blink_timer = 0

label_history = []

# ===== Calibration state =====
calibration_active = False
calibration_step = 0
calibration_samples = []
calibration_complete = False
calibration_mode = 'eye_movement'

# Two-phase calibration:
#   Phase 1: CENTER baseline (30s)
#   Phase 2: LEFT/RIGHT sampling, repeated N times

CENTER_BASELINE_DURATION = 30000  # samples (~30s at 1kHz)
DIRECTION_SAMPLE_DURATION = 5000  # samples per direction (~5s)
DIRECTION_REPETITIONS = 5         # number of LEFT/RIGHT sets

calibration_phase = 'center_baseline'  # 'center_baseline' | 'direction_samples'
direction_sample_count = {'left': 0, 'right': 0}
current_set_number = 0
current_direction = 'left'

eog1_calibration = {
    'center': 0,
    'center_std': 0,
    'center_stability': 0,   # Center dead-zone (absolute ADC units)
    'left_threshold': 0,
    'right_threshold': 0,
    'left_values': [],
    'right_values': [],
    'center_baseline': 0,    # Baseline collected during Phase 1
    'range': 0
}

eog2_calibration = {'center': 0, 'positive_peak': 0, 'negative_peak': 0, 'range': 0}

blink_calibration = {'baseline': 0, 'positive_peak': 0, 'negative_peak': 0, 'range': 0}
blink_samples = []

blink_instructions = [
    "Keep eyes OPEN (relaxed) for 3 seconds",
    "BLINK 5-10 times over 6 seconds"
]

# ===== Serial connection =====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.001)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()


def parse_packet(packet):
    if len(packet) != PACKET_SIZE:
        return None, None, None
    if packet[0] != HEADER[0] or packet[1] != HEADER[1] or packet[2] != HEADER[2]:
        return None, None, None
    adc0 = ((packet[3] & 0x0F) << 8) | packet[4]
    adc1 = ((packet[5] & 0x0F) << 8) | packet[6]
    adc2 = ((packet[7] & 0x0F) << 8) | packet[8]
    return adc0, adc1, adc2


def analyze_emg_frequency(emg_data):
    """Estimate the dominant EMG frequency component via FFT (20-500 Hz band)."""
    if len(emg_data) < 500:
        return 0.0
    emg_array = np.array(emg_data) - np.mean(emg_data)
    window = np.hanning(len(emg_array))
    emg_windowed = emg_array * window
    N = len(emg_windowed)
    yf = fft(emg_windowed)
    xf = fftfreq(N, 1 / SAMPLING_RATE)
    positive_freq_idx = xf > 0
    xf_positive = xf[positive_freq_idx]
    yf_magnitude = np.abs(yf[positive_freq_idx])
    valid_range = (xf_positive >= 20) & (xf_positive <= 500)
    if np.any(valid_range):
        xf_valid = xf_positive[valid_range]
        yf_valid = yf_magnitude[valid_range]
        if len(yf_valid) > 0:
            return xf_valid[np.argmax(yf_valid)]
    return 0.0


def detect_blinks_biphasic(eog2_data, window_size=50):
    """Detect blink events as biphasic (positive-then-negative) peaks in the EOG2 channel."""
    if len(eog2_data) < window_size * 2:
        return {'positive_peaks': [], 'negative_peaks': []}
    eog2_array = np.array(eog2_data)
    baseline = np.median(eog2_array)
    eog2_centered = eog2_array - baseline
    positive_threshold = np.std(eog2_centered) * 1.5
    positive_peaks = []
    for i in range(window_size, len(eog2_centered) - window_size):
        if eog2_centered[i] > positive_threshold:
            is_peak = True
            for j in range(i - 10, i + 10):
                if j != i and 0 <= j < len(eog2_centered):
                    if eog2_centered[j] > eog2_centered[i]:
                        is_peak = False
                        break
            if is_peak:
                positive_peaks.append((i, eog2_centered[i] + baseline))
    negative_threshold = -np.std(eog2_centered) * 1.5
    negative_peaks = []
    for i in range(window_size, len(eog2_centered) - window_size):
        if eog2_centered[i] < negative_threshold:
            is_peak = True
            for j in range(i - 10, i + 10):
                if j != i and 0 <= j < len(eog2_centered):
                    if eog2_centered[j] < eog2_centered[i]:
                        is_peak = False
                        break
            if is_peak:
                negative_peaks.append((i, eog2_centered[i] + baseline))
    return {'positive_peaks': positive_peaks, 'negative_peaks': negative_peaks}


def normalize_eog(eog1_raw, eog2_raw):
    if calibration_mode == 'blink' and blink_calibration['range'] > 0:
        if eog1_calibration['range'] > 0:
            eog1_norm = (eog1_raw - eog1_calibration['center']) / (eog1_calibration['range'] / 2)
        else:
            eog1_norm = 0.0
        eog2_norm = (eog2_raw - blink_calibration['baseline']) / (blink_calibration['range'] / 2)
        return np.clip(eog1_norm, -1.5, 1.5), np.clip(eog2_norm, -1.5, 1.5)

    if eog1_calibration['range'] == 0:
        return 0.0, 0.0

    eog1_norm = (eog1_raw - eog1_calibration['center']) / (eog1_calibration['range'] / 2)
    if calibration_complete and calibration_mode == 'eye_movement':
        eog2_norm = (eog2_raw - eog2_calibration['center']) / (eog1_calibration['range'] / 2)
    else:
        eog2_norm = 0.0

    return np.clip(eog1_norm, -1.5, 1.5), np.clip(eog2_norm, -1.5, 1.5)


def detect_direction_with_center_stability(eog1_raw):
    """
    Classify gaze direction from raw EOG1 using an absolute-deviation
    approach with a dynamic center dead-zone.

    Args:
        eog1_raw: Raw EOG1 ADC value.

    Returns:
        "LEFT", "RIGHT", or "CENTER"
    """
    center = eog1_calibration['center_baseline']
    center_stability = eog1_calibration.get('center_stability', 5.0)

    deviation = eog1_raw - center

    # Step 1: center dead-zone check (highest priority).
    # Within baseline +/- (3 * std), always classify as CENTER.
    if abs(deviation) <= center_stability:
        return "CENTER"

    # Step 2: LEFT/RIGHT threshold check using normalized deviation.
    if eog1_calibration['range'] > 0:
        eog1_norm = deviation / (eog1_calibration['range'] / 2)

        if eog1_norm < -eog1_calibration.get('left_threshold', 0.5):
            return "LEFT"

        if eog1_norm > eog1_calibration.get('right_threshold', 0.5):
            return "RIGHT"

    return "CENTER"


def classify_blink_simple(eog2_normalized):
    """Simple threshold-based blink classification."""
    if abs(eog2_normalized) > BLINK_THRESHOLD:
        return "BLINK"
    else:
        return None


def update_labels_with_persistence(eog1_raw, eog1_norm, eog2_norm):
    """
    Update direction/blink labels with persistence and opposite-direction
    blocking to suppress false transitions.

    Args:
        eog1_raw: Raw EOG1 ADC value.
        eog1_norm: Normalized EOG1 value.
        eog2_norm: Normalized EOG2 value.

    Logic:
        1. Default label is CENTER.
        2. LEFT detected -> block RIGHT for OPPOSITE_DIRECTION_BLOCK_FRAMES.
        3. RIGHT detected -> block LEFT for OPPOSITE_DIRECTION_BLOCK_FRAMES.
        4. While blocked, the opposite direction is ignored (forced to CENTER).
        5. After the hold timer expires, a short transition-ignore window
           forces CENTER before new directions are accepted again.
    """
    global current_direction_label, last_detected_direction, direction_timer
    global transition_ignore_counter, opposite_block_counter, blocked_direction
    global current_blink_label, blink_timer
    global label_history

    # ========== Direction (uses raw value) ==========
    detected_direction = detect_direction_with_center_stability(eog1_raw)

    # Decrement opposite-direction block counter
    if opposite_block_counter > 0:
        opposite_block_counter -= 1

        if detected_direction == blocked_direction:
            if DEBUG_MODE and opposite_block_counter % 10 == 0:
                print(f"Blocking {blocked_direction} detection ({opposite_block_counter} frames left)")
            detected_direction = "CENTER"  # Force to CENTER while blocked

        if opposite_block_counter == 0:
            if DEBUG_MODE:
                print(f"Unblocking {blocked_direction}")
            blocked_direction = None

    # Decrement transition-ignore counter
    if transition_ignore_counter > 0:
        transition_ignore_counter -= 1
        # Force CENTER during the ignore window
        if detected_direction == "CENTER":
            current_direction_label = "CENTER"
            if DEBUG_MODE and transition_ignore_counter % 5 == 0:
                print(f"Transition ignore: {transition_ignore_counter} frames left")

    elif detected_direction in ["LEFT", "RIGHT"]:
        if detected_direction != last_detected_direction:
            # New direction detected
            last_detected_direction = detected_direction
            direction_timer = DIRECTION_PERSISTENCE_FRAMES
            current_direction_label = detected_direction

            if detected_direction == "LEFT":
                opposite_block_counter = OPPOSITE_DIRECTION_BLOCK_FRAMES
                blocked_direction = "RIGHT"
                if DEBUG_MODE:
                    print(f"LEFT detected. Blocking RIGHT for {OPPOSITE_DIRECTION_BLOCK_FRAMES/20:.1f}s")
            elif detected_direction == "RIGHT":
                opposite_block_counter = OPPOSITE_DIRECTION_BLOCK_FRAMES
                blocked_direction = "LEFT"
                if DEBUG_MODE:
                    print(f"RIGHT detected. Blocking LEFT for {OPPOSITE_DIRECTION_BLOCK_FRAMES/20:.1f}s")
        else:
            # Same direction: just refresh the hold timer
            direction_timer = DIRECTION_PERSISTENCE_FRAMES
            current_direction_label = detected_direction

    elif direction_timer > 0:
        # Hold timer still active: keep the previous direction
        direction_timer -= 1
        current_direction_label = last_detected_direction

        if direction_timer == 0:
            transition_ignore_counter = TRANSITION_IGNORE_FRAMES
            if DEBUG_MODE:
                print(f"Timer expired. Starting transition ignore ({TRANSITION_IGNORE_FRAMES} frames)")

        if DEBUG_MODE and direction_timer % 20 == 0:
            print(f"{current_direction_label} holding... {direction_timer/20:.1f}s left")

    else:
        # Hold timer and ignore window both expired: return to CENTER
        if detected_direction == "CENTER":
            current_direction_label = "CENTER"

    # ========== Blink (2s hold) ==========
    detected_blink = classify_blink_simple(eog2_norm)

    if detected_blink == "BLINK":
        blink_timer = BLINK_PERSISTENCE_FRAMES
        current_blink_label = "BLINK"
    elif blink_timer > 0:
        blink_timer -= 1
        current_blink_label = "BLINK"
    else:
        current_blink_label = "NO_BLINK"

    # ========== History ==========
    label_history.append({
        'timestamp': time.time(),
        'direction': current_direction_label,
        'blink': current_blink_label,
        'eog1_raw': eog1_raw,
        'eog1_norm': eog1_norm,
        'eog2_norm': eog2_norm
    })

    if len(label_history) > 10000:
        label_history.pop(0)


def get_current_labels():
    return {'direction': current_direction_label, 'blink': current_blink_label}


def export_label_history(filename='label_history.csv'):
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'direction', 'blink', 'eog1_raw', 'eog1_norm', 'eog2_norm'])
        writer.writeheader()
        writer.writerows(label_history)
    print(f"\nLabel history exported to {filename}")
    print(f"  Total records: {len(label_history)}")


def smart_scale_axis(ax, data, signal_name, margin_ratio=0.15):
    if len(data) == 0:
        return
    y_min_data = min(data)
    y_max_data = max(data)
    data_range = y_max_data - y_min_data
    ranges = SMART_Y_RANGES.get(signal_name, {'min': 100, 'max': 4095})
    if data_range < ranges['min']:
        center = (y_min_data + y_max_data) / 2
        y_min_target = center - ranges['min'] / 2
        y_max_target = center + ranges['min'] / 2
    elif data_range > ranges['max']:
        center = (y_min_data + y_max_data) / 2
        y_min_target = center - ranges['max'] / 2
        y_max_target = center + ranges['max'] / 2
    else:
        margin = data_range * margin_ratio
        y_min_target = y_min_data - margin
        y_max_target = y_max_data + margin
    if Y_AXIS_SMOOTHING > 0:
        current_min, current_max = ax.get_ylim()
        alpha = 1 - Y_AXIS_SMOOTHING
        y_min_smooth = alpha * y_min_target + (1 - alpha) * current_min
        y_max_smooth = alpha * y_max_target + (1 - alpha) * current_max
    else:
        y_min_smooth = y_min_target
        y_max_smooth = y_max_target
    ax.set_ylim(y_min_smooth, y_max_smooth)


def read_serial_data():
    global sample_count, calibration_active, calibration_step, calibration_samples
    global blink_samples, dominant_freq
    buffer = bytearray()
    packets_processed = 0
    max_packets_per_update = 200
    bytes_waiting = ser.in_waiting
    max_allowed_bytes = max_packets_per_update * PACKET_SIZE * 1.5
    if bytes_waiting > max_allowed_bytes:
        discard_bytes = int(bytes_waiting - max_allowed_bytes)
        ser.read(discard_bytes)
        bytes_waiting = ser.in_waiting
    if bytes_waiting > 0:
        buffer.extend(ser.read(bytes_waiting))
    i = 0
    while i < len(buffer) - PACKET_SIZE + 1 and packets_processed < max_packets_per_update:
        if buffer[i] == HEADER[0] and buffer[i + 1] == HEADER[1] and buffer[i + 2] == HEADER[2]:
            packet = buffer[i:i + PACKET_SIZE]
            if len(packet) == PACKET_SIZE:
                eog2, eog1, emg = parse_packet(packet)
                if emg is not None:
                    emg_buffer.append(emg)
                    eog1_buffer.append(eog1)
                    eog2_buffer.append(eog2)
                    emg_fft_buffer.append(emg)
                    if DEBUG_MODE and sample_count % 100 == 0:
                        print(f"[DEBUG] Raw: EOG1={eog1:.0f}, EOG2={eog2:.0f}, EMG={emg:.0f}")
                    eog1_norm, eog2_norm = normalize_eog(eog1, eog2)
                    eog1_normalized_buffer.append(eog1_norm)
                    eog2_normalized_buffer.append(eog2_norm)
                    if calibration_complete:
                        update_labels_with_persistence(eog1, eog1_norm, eog2_norm)
                    sample_count += 1
                    if sample_count % 500 == 0 and len(emg_fft_buffer) >= 500:
                        dominant_freq = analyze_emg_frequency(list(emg_fft_buffer))
                    if calibration_active:
                        if calibration_mode == 'eye_movement':
                            calibration_samples.append((eog1, eog2))

                            # Phase 1: CENTER baseline
                            if calibration_phase == 'center_baseline':
                                if len(calibration_samples) >= CENTER_BASELINE_DURATION:
                                    process_calibration_step()

                            # Phase 2: direction samples
                            elif calibration_phase == 'direction_samples':
                                if len(calibration_samples) >= DIRECTION_SAMPLE_DURATION:
                                    process_calibration_step()

                        elif calibration_mode == 'blink':
                            blink_samples.append((eog1, eog2))
                            if calibration_step == 0 and len(blink_samples) >= 3000:
                                process_blink_baseline()
                            elif calibration_step == 1 and len(blink_samples) >= 6000:
                                process_blink_detection()
                packets_processed += 1
                i += PACKET_SIZE
            else:
                i += 1
        else:
            i += 1


def process_blink_baseline():
    global calibration_step, blink_samples, blink_calibration
    eog2_values = [s[1] for s in blink_samples]
    baseline = np.median(eog2_values)
    blink_calibration['baseline'] = baseline
    print(f"\nBlink calibration - baseline: {baseline:.2f} +/- {np.std(eog2_values):.2f}")
    blink_samples.clear()
    calibration_step += 1
    status_text.set_text(f"Step 2/2: {blink_instructions[1]}")


def process_blink_detection():
    global calibration_active, blink_samples, blink_calibration, calibration_complete
    eog2_values = [s[1] for s in blink_samples]
    blinks = detect_blinks_biphasic(eog2_values, window_size=30)
    if len(blinks['positive_peaks']) == 0 or len(blinks['negative_peaks']) == 0:
        print("\nInsufficient blinks detected.")
        status_text.set_text("Insufficient blinks. Retry.")
        calibration_active = False
        blink_samples.clear()
        return
    max_positive = max([p[1] for p in blinks['positive_peaks']])
    min_negative = min([p[1] for p in blinks['negative_peaks']])
    blink_calibration['positive_peak'] = max_positive
    blink_calibration['negative_peak'] = min_negative
    blink_calibration['range'] = abs(max_positive - min_negative)
    print("\nBlink calibration complete.")
    print(f"  Range: {blink_calibration['range']:.2f}")
    calibration_active = False
    calibration_complete = True
    blink_samples.clear()
    status_text.set_text("Blink Calibration Complete!")


def process_calibration_step():
    """Process one step of the two-phase calibration (center baseline, then LEFT/RIGHT sets)."""
    global calibration_active, calibration_step, calibration_samples, calibration_phase
    global eog1_calibration, eog2_calibration, calibration_complete, direction_sample_count
    global current_set_number, current_direction

    if len(calibration_samples) == 0:
        return

    eog1_values = [s[0] for s in calibration_samples]
    eog2_values = [s[1] for s in calibration_samples]
    eog1_avg = np.mean(eog1_values)
    eog1_std = np.std(eog1_values)
    eog2_avg = np.mean(eog2_values)

    # Phase 1: collect CENTER baseline (30s)
    if calibration_phase == 'center_baseline':
        print("\nCENTER baseline complete.")
        print(f"  EOG1: {eog1_avg:.2f} +/- {eog1_std:.2f}")
        print(f"  Samples collected: {len(eog1_values)}")

        eog1_calibration['center_baseline'] = eog1_avg
        eog1_calibration['center'] = eog1_avg
        eog1_calibration['center_std'] = eog1_std
        eog2_calibration['center'] = eog2_avg

        # Move to Phase 2
        calibration_phase = 'direction_samples'
        calibration_samples.clear()
        direction_sample_count = {'left': 0, 'right': 0}
        current_set_number = 1
        current_direction = 'left'

        status_text.set_text(f"Set {current_set_number}/{DIRECTION_REPETITIONS}: Look LEFT")
        print("\n" + "=" * 70)
        print("Phase 2: Direction Sampling (LEFT-RIGHT Sets)")
        print("=" * 70)
        print(f"Total sets: {DIRECTION_REPETITIONS}")
        print(f"Each set: LEFT ({DIRECTION_SAMPLE_DURATION/1000:.0f}s) -> RIGHT ({DIRECTION_SAMPLE_DURATION/1000:.0f}s)")

    # Phase 2: LEFT/RIGHT set sampling
    elif calibration_phase == 'direction_samples':
        deviation = eog1_avg - eog1_calibration['center_baseline']

        if current_direction == 'left':
            direction_sample_count['left'] += 1
            eog1_calibration['left_values'].append(eog1_avg)
            print(f"\nSet {current_set_number}/{DIRECTION_REPETITIONS} - LEFT")
            print(f"  EOG1: {eog1_avg:.2f} (deviation: {deviation:.2f})")

            current_direction = 'right'
            status_text.set_text(f"Set {current_set_number}/{DIRECTION_REPETITIONS}: Look RIGHT")

        elif current_direction == 'right':
            direction_sample_count['right'] += 1
            eog1_calibration['right_values'].append(eog1_avg)
            print(f"\nSet {current_set_number}/{DIRECTION_REPETITIONS} - RIGHT")
            print(f"  EOG1: {eog1_avg:.2f} (deviation: {deviation:.2f})")

            current_set_number += 1

            if current_set_number <= DIRECTION_REPETITIONS:
                current_direction = 'left'
                status_text.set_text(f"Set {current_set_number}/{DIRECTION_REPETITIONS}: Look LEFT")

        calibration_samples.clear()

        # All sets complete
        if (direction_sample_count['left'] >= DIRECTION_REPETITIONS and
                direction_sample_count['right'] >= DIRECTION_REPETITIONS):
            finish_calibration()
            calibration_active = False
            calibration_complete = True
            status_text.set_text("Eye Calibration Complete!")


def finish_calibration():
    """Compute final thresholds and dead-zone from the collected calibration data."""
    global eog1_calibration

    left_mean = np.mean(eog1_calibration['left_values'])
    left_std = np.std(eog1_calibration['left_values'])
    right_mean = np.mean(eog1_calibration['right_values'])
    right_std = np.std(eog1_calibration['right_values'])

    center = eog1_calibration['center_baseline']
    center_std = eog1_calibration['center_std']

    eog1_calibration['range'] = abs(right_mean - left_mean)

    left_deviation = abs(left_mean - center)
    right_deviation = abs(right_mean - center)

    # Minimum absolute threshold to guard against a very small range
    MIN_ABSOLUTE_THRESHOLD = 8.0  # ADC units

    # Use 70% of the observed deviation as the threshold (require a clear movement)
    left_threshold_abs = max(left_deviation * 0.7, MIN_ABSOLUTE_THRESHOLD)
    right_threshold_abs = max(right_deviation * 0.7, MIN_ABSOLUTE_THRESHOLD)

    if eog1_calibration['range'] > 0:
        eog1_calibration['left_threshold'] = left_threshold_abs / (eog1_calibration['range'] / 2)
        eog1_calibration['right_threshold'] = right_threshold_abs / (eog1_calibration['range'] / 2)
    else:
        # Range is zero: set an effectively unreachable threshold
        eog1_calibration['left_threshold'] = 10.0
        eog1_calibration['right_threshold'] = 10.0

    # Center dead-zone: 3x the baseline standard deviation, minimum 5 ADC units
    center_stability_abs = max(center_std * 3, 5.0)
    eog1_calibration['center_stability'] = center_stability_abs

    print("\n" + "=" * 70)
    print("EYE CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"CENTER baseline: {center:.2f} +/- {center_std:.2f}")
    print(f"  Stability zone: +/-{center_stability_abs:.2f} ADC")
    print(f"\nLEFT:  {left_mean:.2f} +/- {left_std:.2f}")
    print(f"       Deviation from center: {left_mean - center:.2f}")
    print(f"       Threshold: {left_threshold_abs:.2f} ADC")
    print(f"\nRIGHT: {right_mean:.2f} +/- {right_std:.2f}")
    print(f"       Deviation from center: {right_mean - center:.2f}")
    print(f"       Threshold: {right_threshold_abs:.2f} ADC")
    print(f"\nTotal range: {eog1_calibration['range']:.2f}")
    print("\nDetection logic:")
    print(f"  CENTER: {center - center_stability_abs:.2f} ~ {center + center_stability_abs:.2f}")
    print(f"  LEFT:   < {center - left_threshold_abs:.2f}")
    print(f"  RIGHT:  > {center + right_threshold_abs:.2f}")
    print("\nNormalized thresholds:")
    print(f"  LEFT:  < -{eog1_calibration['left_threshold']:.3f}")
    print(f"  RIGHT: > +{eog1_calibration['right_threshold']:.3f}")
    print("\nPersistence settings:")
    print(f"  Direction: {DIRECTION_PERSISTENCE_SECONDS} seconds")
    print(f"  Blink: {BLINK_PERSISTENCE_SECONDS} seconds")
    print(f"  Transition ignore: {TRANSITION_IGNORE_FRAMES} frames")
    print(f"  Opposite-direction block: {OPPOSITE_DIRECTION_BLOCK_FRAMES} frames ({OPPOSITE_DIRECTION_BLOCK_FRAMES/20:.1f}s)")
    print("=" * 70)


def start_calibration(event):
    global calibration_active, calibration_step, calibration_samples, calibration_mode
    global eog1_calibration, calibration_phase, direction_sample_count
    global current_set_number, current_direction

    if calibration_active:
        return

    eog1_calibration['left_values'] = []
    eog1_calibration['right_values'] = []
    eog1_calibration['center_baseline'] = 0

    calibration_mode = 'eye_movement'
    calibration_active = True
    calibration_phase = 'center_baseline'
    calibration_step = 0
    calibration_samples.clear()
    direction_sample_count = {'left': 0, 'right': 0}
    current_set_number = 0
    current_direction = 'left'

    status_text.set_text(f"Phase 1: Look at CENTER for {CENTER_BASELINE_DURATION/1000:.0f}s")

    print("\n" + "=" * 70)
    print("TWO-PHASE EYE CALIBRATION")
    print("=" * 70)
    print("Phase 1: CENTER Baseline")
    print(f"  Duration: {CENTER_BASELINE_DURATION/1000:.0f} seconds")
    print("  Purpose: establish a stable center reference")
    print("\nPhase 2: Direction Sampling (LEFT-RIGHT Sets)")
    print(f"  Total sets: {DIRECTION_REPETITIONS}")
    print(f"  Each set: LEFT ({DIRECTION_SAMPLE_DURATION/1000:.0f}s) -> RIGHT ({DIRECTION_SAMPLE_DURATION/1000:.0f}s)")
    print(f"  Total samples: {DIRECTION_REPETITIONS * 2} (LEFT: {DIRECTION_REPETITIONS}, RIGHT: {DIRECTION_REPETITIONS})")
    print(f"\nTotal time: ~{(CENTER_BASELINE_DURATION + DIRECTION_SAMPLE_DURATION * DIRECTION_REPETITIONS * 2) / 1000:.0f} seconds")
    print("=" * 70)


def start_blink_calibration(event):
    global calibration_active, calibration_step, blink_samples, calibration_mode
    if calibration_active:
        return
    calibration_mode = 'blink'
    calibration_active = True
    calibration_step = 0
    blink_samples.clear()
    status_text.set_text(f"Step 1/2: {blink_instructions[0]}")
    print("\n" + "=" * 70)
    print("BLINK CALIBRATION STARTED")
    print("=" * 70)


# ===== Visualization setup =====
plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('#0a0a0a')
gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 3, 3, 3], width_ratios=[3, 0.8, 3, 0.8],
                       hspace=0.35, wspace=0.15, left=0.06, right=0.97, top=0.94, bottom=0.08)

ax_status = fig.add_subplot(gs[0, :2])
ax_status.axis('off')
status_text = ax_status.text(0.5, 0.5, "Press 'Eye Cal' or 'Blink Cal'", ha='center', va='center', fontsize=13,
                              color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.8', facecolor='#4a148c',
                                       edgecolor='#7b1fa2', linewidth=2, alpha=0.9))

ax_freq = fig.add_subplot(gs[0, 2:])
ax_freq.axis('off')
ax_freq.set_xlim(0, 1)
ax_freq.set_ylim(0, 1)
freq_circle = Circle((0.5, 0.5), 0.35, fill=False, edgecolor='#00bcd4', linewidth=8, alpha=0.3)
ax_freq.add_patch(freq_circle)
freq_arc = None
freq_text = ax_freq.text(0.5, 0.5, '0', ha='center', va='center', fontsize=28, color='#00bcd4', weight='bold')
freq_label = ax_freq.text(0.5, 0.15, 'Hz', ha='center', va='center', fontsize=11, color='#80deea', alpha=0.8)
freq_title = ax_freq.text(0.5, 0.88, 'EMG Freq', ha='center', va='center', fontsize=10, color='#80deea', weight='bold')

ax_emg = fig.add_subplot(gs[1, :])
ax_eog1 = fig.add_subplot(gs[2, :2])
ax_dir_label = fig.add_subplot(gs[2, 2:])
ax_dir_label.axis('off')
ax_dir_label.set_xlim(0, 1)
ax_dir_label.set_ylim(0, 1)
dir_bg = FancyBboxPatch((0.1, 0.25), 0.8, 0.5, boxstyle="round,pad=0.1", facecolor='#1a1a1a',
                         edgecolor='#69f0ae', linewidth=3, alpha=0.9)
ax_dir_label.add_patch(dir_bg)
dir_title = ax_dir_label.text(0.5, 0.85, 'Direction', ha='center', va='center', fontsize=11, color='#69f0ae', weight='bold')
dir_text = ax_dir_label.text(0.5, 0.5, 'CENTER', ha='center', va='center', fontsize=22, color='#69f0ae', weight='bold')

ax_eog2 = fig.add_subplot(gs[3, :2])
ax_blink_label = fig.add_subplot(gs[3, 2:])
ax_blink_label.axis('off')
ax_blink_label.set_xlim(0, 1)
ax_blink_label.set_ylim(0, 1)
blink_bg = FancyBboxPatch((0.1, 0.25), 0.8, 0.5, boxstyle="round,pad=0.1", facecolor='#1a1a1a',
                           edgecolor='#448aff', linewidth=3, alpha=0.9)
ax_blink_label.add_patch(blink_bg)
blink_title = ax_blink_label.text(0.5, 0.85, 'Blink', ha='center', va='center', fontsize=11, color='#448aff', weight='bold')
blink_text = ax_blink_label.text(0.5, 0.5, 'NO_BLINK', ha='center', va='center', fontsize=18, color='#448aff', weight='bold')

fig.suptitle('EOG/EMG System: Opposite-Direction Blocking',
             fontsize=16, fontweight='bold', color='#00bcd4', y=0.98)

line_emg, = ax_emg.plot([], [], color='#ff5252', linewidth=1.2, label='EMG', alpha=0.9)
line_eog1, = ax_eog1.plot([], [], color='#69f0ae', linewidth=1.2, label='EOG1', alpha=0.9)
line_eog2, = ax_eog2.plot([], [], color='#448aff', linewidth=1.2, label='EOG2', alpha=0.9)

for ax, name, color, light_color in [(ax_emg, 'EMG Signal', '#ff5252', '#ff8a80'),
                                      (ax_eog1, 'EOG1 - Horizontal', '#69f0ae', '#b9f6ca'),
                                      (ax_eog2, 'EOG2 - Vertical', '#448aff', '#82b1ff')]:
    ax.set_xlim(0, DISPLAY_SAMPLES)
    ax.set_ylim(0, 4095)
    ax.set_ylabel('ADC', fontsize=10, color=light_color, weight='bold')
    ax.set_title(name, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.3, fontsize=9)
    ax.set_facecolor('#0f0f0f')
    ax.spines['top'].set_color(color)
    ax.spines['top'].set_linewidth(2)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_color('#333333')
    ax.tick_params(colors='#888888', labelsize=8)

ax_eog2.set_xlabel('Samples', fontsize=10, color='#888888')

btn_ax1 = fig.add_axes([0.30, 0.02, 0.15, 0.035])
btn_eye = Button(btn_ax1, 'Eye Cal', color='#4a148c', hovercolor='#6a1b9a')
btn_eye.label.set_color('white')
btn_eye.label.set_weight('bold')
btn_eye.on_clicked(start_calibration)

btn_ax2 = fig.add_axes([0.55, 0.02, 0.15, 0.035])
btn_blink = Button(btn_ax2, 'Blink Cal', color='#00695c', hovercolor='#00897b')
btn_blink.label.set_color('white')
btn_blink.label.set_weight('bold')
btn_blink.on_clicked(start_blink_calibration)


def update(frame):
    global freq_arc
    read_serial_data()
    if len(emg_buffer) > 0:
        x_data = np.arange(len(emg_buffer))
        line_emg.set_data(x_data, list(emg_buffer))
        smart_scale_axis(ax_emg, list(emg_buffer), 'emg')
    if dominant_freq > 0:
        freq_text.set_text(f'{int(dominant_freq)}')
        if freq_arc:
            freq_arc.remove()
        angle = min(dominant_freq / 500 * 270, 270)
        theta = np.linspace(90, 90 + angle, 100)
        x = 0.5 + 0.35 * np.cos(np.radians(theta))
        y = 0.5 + 0.35 * np.sin(np.radians(theta))
        freq_arc, = ax_freq.plot(x, y, color='#00bcd4', linewidth=8, solid_capstyle='round')

    if calibration_complete:
        dir_text.set_text(current_direction_label)
        dir_colors = {'LEFT': '#ff9800', 'CENTER': '#69f0ae', 'RIGHT': '#2196f3'}
        dir_text.set_color(dir_colors.get(current_direction_label, '#69f0ae'))
        dir_bg.set_edgecolor(dir_colors.get(current_direction_label, '#69f0ae'))

        blink_text.set_text(current_blink_label)
        if current_blink_label == 'BLINK':
            blink_text.set_color('#ff5252')
            blink_bg.set_edgecolor('#ff5252')
        else:
            blink_text.set_color('#448aff')
            blink_bg.set_edgecolor('#448aff')

    if len(eog1_buffer) > 0:
        x_data = np.arange(len(eog1_buffer))
        line_eog1.set_data(x_data, list(eog1_buffer))
        smart_scale_axis(ax_eog1, list(eog1_buffer), 'eog1')

    if len(eog2_buffer) > 0:
        x_data = np.arange(len(eog2_buffer))
        line_eog2.set_data(x_data, list(eog2_buffer))
        smart_scale_axis(ax_eog2, list(eog2_buffer), 'eog2')

    return line_emg, line_eog1, line_eog2


ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False, cache_frame_data=False)

print("\n" + "=" * 70)
print("EOG/EMG DETECTION SYSTEM")
print("=" * 70)
print("Repeated calibration (5x)")
print("   CENTER -> LEFT -> CENTER -> RIGHT -> CENTER (x5)")
print("   Robust baseline and threshold calculation")
print("")
print("Stable center detection")
print(f"   Center stability zone: +/-{CENTER_STABILITY_THRESHOLD}")
print("   Adaptive thresholds based on calibration")
print("")
print("Transition filter")
print(f"   Ignores {TRANSITION_IGNORE_FRAMES} frames after timer expiry")
print("   Prevents false detections during return")
print("")
print("Opposite-direction blocking")
print(f"   LEFT detected -> blocks RIGHT for {OPPOSITE_DIRECTION_BLOCK_FRAMES/20:.1f}s")
print(f"   RIGHT detected -> blocks LEFT for {OPPOSITE_DIRECTION_BLOCK_FRAMES/20:.1f}s")
print("   Prevents LEFT->CENTER->RIGHT and RIGHT->CENTER->LEFT misdetection")
print("")
print("Extended blink display")
print(f"   Blink persistence: {BLINK_PERSISTENCE_SECONDS} seconds")
print("")
print("=" * 70)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nStopping...")
    save = input("Save label history? (y/n): ")
    if save.lower() == 'y':
        export_label_history()
finally:
    ser.close()
    print("Serial port closed.")
