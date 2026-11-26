"""
Runs wavlex_v1.py on N_WORKERS slices of OBSIDS in parallel.
Only needed when N_WORKERS > 1, otherwise just run `python wavlex_v1.py`.
N_WORKERS here must match the value in wavlex_v1.py.
"""

import subprocess

N_WORKERS = 1           # same as N_WORKERS in wavlex_v1.py

processes = []

for i in range(1, N_WORKERS + 1):
    p = subprocess.Popen(
        ["python", "wavlex_v1.py", str(i)]
    )
    processes.append(p)

for p in processes:
    p.wait()
