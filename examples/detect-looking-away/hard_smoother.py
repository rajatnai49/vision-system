from collections import deque

class HardSmoother:
    def __init__(self, window_size=30, threshold=0.69) -> None:
        self.window_size = window_size
        self.threshold = threshold

        self.buffer = deque(maxlen=window_size)

    def update(self, looking_away:bool) -> bool:
        self.buffer.append(looking_away)

        if len(self.buffer) < self.window_size:
            return False

        avg = sum(self.buffer) / len(self.buffer)

        if avg >= self.threshold:
            return True

        return False

