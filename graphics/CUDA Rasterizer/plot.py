"""CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   plot.py ------------------------------------------------------------------"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Check if filename is provided as command line argument
if len(sys.argv) < 2:
    print("Provide filename as command line argument")
    sys.exit(1)
filename = sys.argv[1]

# Read file
with open(filename, 'r') as file:
    lines = file.readlines()

# Process data
frames = []
measurements = []
for line in lines:
    data = line.strip().split(' ')
    frame = int(data[0])
    measurements.append([int(x) for x in data[1:]])
    frames.append(frame)

# Calculate average time and standard deviation
averages = np.mean(measurements, axis=0)
std_deviations = np.std(measurements, axis=0)
relative_percentages = (std_deviations / averages) * 100

# Output average time and standard deviation to terminal
num_measurements = len(measurements[0])
labels = ['Total', 'Scene Fetch', 'Vertex Shader', 'Primitive Assembly',
          'Rasterization', 'Pixel Shader', 'Raster Operation']
for i in range(num_measurements):
    print(f"{labels[i]}:")
    print(f"Average Time: {averages[i]:.2f} us")
    print(f"Standard Deviation: {std_deviations[i]:.2f} us")
    print(f"Relative Percentage: {relative_percentages[i]:.2f}%")
    print()

# Plot data
for i in range(num_measurements):
    if labels[i] == 'Total':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='black')
    elif labels[i] == 'Scene Fetch':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#86A6E1', linestyle='dotted')
    elif labels[i] == 'Vertex Shader':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#CC6D4B')
    elif labels[i] == 'Primitive Assembly':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#DDB9A8')
    elif labels[i] == 'Rasterization':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#86A6E1')
    elif labels[i] == 'Pixel Shader':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#AFE0AA')
    elif labels[i] == 'Raster Operation':
        plt.plot(frames, [m[i] for m in measurements], label=labels[i],
            color='#4BCC4B', linestyle='dotted')

plt.xlabel('Frame Count')
plt.ylabel('Time (us)')
plt.legend()
#plt.yscale('log')
plt.show()
