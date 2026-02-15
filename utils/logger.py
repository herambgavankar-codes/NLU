import time

def log(message):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {message}")
