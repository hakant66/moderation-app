# backend/rate_limit.py
import time, threading
class TokenBucket:
    def __init__(self, rate_per_sec=2, capacity=4):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()
    def acquire(self, tokens=1, block=True):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts)*self.rate)
                self.ts = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            if not block:
                return False
            time.sleep(0.05)
bucket = TokenBucket(rate_per_sec=2, capacity=4)  # tune to your limit
