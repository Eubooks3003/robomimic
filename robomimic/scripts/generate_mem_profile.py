import matplotlib.pyplot as plt

def read_memory_profile(filename):
    times = []
    memory = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('MEM'):
                parts = line.split()
                times.append(float(parts[1]))
                memory.append(float(parts[2]))

    return times, memory

# Usage
times, memory = read_memory_profile('mprofile_20240703154528.dat')
plt.plot(times, memory)
plt.title('Memory Usage Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Memory (MB)')

# Save the plot to a file
plt.savefig('memory_usage_plot.png')
# If you still want to display it after saving, you can uncomment the next line
# plt.show()
