import time

accumulated_times = {}
def start_timing(label):
    if label not in accumulated_times:
        accumulated_times[label] = {'start_time': 0, 'total_time': 0}
    accumulated_times[label]['start_time'] = time.perf_counter()

def stop_timing(label):
    if label in accumulated_times and accumulated_times[label]['start_time'] != 0:
        end_time = time.perf_counter()
        elapsed_time = end_time - accumulated_times[label]['start_time']
        accumulated_times[label]['total_time'] += elapsed_time
        accumulated_times[label]['start_time'] = 0  # Reset start time

def print_times():
    for label, times in accumulated_times.items():
        print(f"{label}: {times['total_time']:.4f} seconds")