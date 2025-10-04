# smoothing.py
class EMAFilter:
    def __init__(self, alpha=0.25): self.a = alpha; self.prev = None
    def __call__(self, xy):
        if self.prev is None: self.prev = xy; return xy
        self.prev = (self.a*xy[0]+(1-self.a)*self.prev[0], self.a*xy[1]+(1-self.a)*self.prev[1])
        return self.prev
