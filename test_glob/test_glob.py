import os
from glob import glob

print(glob('*.txt'))

print(sorted(glob('*.txt')))

size_low_to_high = sorted(glob('*.txt'), key=os.path.getsize)
print(size_low_to_high)

time_past_to_now = sorted(glob('*.txt'), key=os.path.getmtime)
print(time_past_to_now)