"""
1. Enable NeurOne Digital Out Feature
Bittium NeurOne's hardware-based Digital Out module provides deterministic real-time streaming with constant latency ():

Configure Digital Out via NeurOne's GUI (Settings > Digital Out)

Select EMG channels and sampling rate (matches primary acquisition settings)

Streaming uses UDP protocol for network transmission

2. Network Configuration
NeurOne streams data via User Datagram Protocol (UDP). Configure these network parameters:

"""
# Python UDP receiver setup example
import socket

UDP_IP = "192.168.1.100"   # NeurOne PC IP
UDP_PORT = 12345           # Default NeurOne Digital Out port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


"""
3. Data Packet Structure
NeurOne Digital Out sends data in binary packets containing:

8-byte timestamp (double precision)

4-byte channel count (int32)

N x 4-byte float values (EMG samples)

"""
import struct

def parse_packet(data):
    timestamp = struct.unpack('d', data[0:8])[0]
    channel_count = struct.unpack('i', data[8:12])[0]
    samples = struct.unpack('f'*channel_count, data[12:])
    return timestamp, samples

"""
4. Real-Time Plotting Implementation
Use matplotlib animation for continuous updating:

"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Circular buffer for 5s data at 1kHz
BUFFER_SIZE = 5000
emg_buffer = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

def update_plot(frame):
    data, _ = sock.recvfrom(4096)  # Max packet size
    _, samples = parse_packet(data)
    emg_buffer.extend(samples)
    line.set_ydata(emg_buffer)
    return line,

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim(-0.1, 0.1)            # Typical EMG range (V)
line, = ax.plot(emg_buffer)
ani = animation.FuncAnimation(fig, update_plot, interval=10)
plt.show()

"""
5. Key Considerations
Latency: Hardware-based streaming ensures ~2ms constant latency ()

Synchronization: Use embedded timestamps for multi-system integration

Data Integrity: Always save raw data simultaneously via NeurOne's primary acquisition ()

Filtering: Implement real-time bandpass (20-450Hz) using scipy.signal:

"""

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(20, 450, 1000)
filtered_emg = lfilter(b, a, samples)

"""
Alternative Approach: LSL Integration
For Lab Streaming Layer compatibility ():

Install LSL bindings: pip install pylsl

Create LSL outlet in NeurOne's API

Python receiver code:

"""
from pylsl import resolve_stream, StreamInlet

streams = resolve_stream('type', 'EMG')
inlet = StreamInlet(streams[0])

while True:
    sample, timestamp = inlet.pull_sample()
    # Process sample

# ------------------------------

# Investigate the structure of the binary data
import struct

file_path = '/home/victormoraes/Downloads/Temp/bittium_file_5triggers.txt'

with open(file_path, 'rb') as f:
    data = f.read(64)  # Read first 64 bytes

# Print byte offsets and values
for i in range(0, len(data), 4):
    chunk = data[i:i+4]
    print(f"Bytes {i:02}-{i+3:02}: {chunk.hex(' ')}")


print(struct.unpack('d', data[0:8]))  # Double precision
print(struct.unpack('f', data[0:4]))  # Single precision


import binascii


with open(file_path, 'r') as f:
    hex_string = f.read()  # Read the entire file content

# Remove spaces and '\\x' prefixes, then convert to bytes
hex_string = hex_string.replace(" ", "").replace("\\x", "")
binary_data = binascii.unhexlify(hex_string)

# Now you have the binary data, you can parse it
# Example: print the first few bytes
print(binary_data[:20])














# Open binary file
file_path = '/home/victormoraes/Downloads/Temp/bittium_file_5triggers.txt'
with open(file_path, 'rb') as f:
    data = f.read()

import struct

def parse_packet(data):
    timestamp = struct.unpack('d', data[0:8])[0]
    channel_count = struct.unpack('i', data[8:12])[0]
    samples = struct.unpack('f'*channel_count, data[12:])
    return timestamp, samples

channel_count = 1  # Set the correct channel count
packet_size = 8 + 4 + (channel_count * 4)

all_timestamps = []
all_samples = []

with open(file_path, 'rb') as f: # f is a BufferedReader object
    while True:
        packet = f.read(packet_size) # Read a chunk of binary data
        if not packet:
            break  # End of file
        if len(packet) != packet_size:
            print("Warning: incomplete packet")
            break

        timestamp, samples = parse_packet(packet)
        all_timestamps.append(timestamp)
        all_samples.append(samples)

print("Number of timestamps:", len(all_timestamps))
print("Number of sample sets:", len(all_samples))

# Convert to numpy arrays
timestamps = np.array(all_timestamps)
samples = np.array(all_samples)
