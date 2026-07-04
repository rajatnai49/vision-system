class EmaSmoother:
    def __init__(self, alpha=0.1, threshold=0.69) -> None:
        self.alpha = alpha
        self.threshold = threshold

        self.ema = None

    def update(self, looking_away: bool) -> bool:
        current_result = 1.0 if looking_away else 0.0

        if self.ema is None:
            self.ema = current_result
            return looking_away

        self.ema = self.alpha * current_result + ((1.0 - self.alpha) * self.ema)

        return self.ema >= self.threshold

