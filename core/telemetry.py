import threading, queue, csv, time

class TelemetryLogger:
    def __init__(self, path):
        self.q = queue.Queue(maxsize=5000)
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.path = path

    def start(self):
        self.t.start()

    def _worker(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts","id","cls","cx","cy","theta_x_deg","theta_y_deg","omega_x_deg_s","omega_y_deg_s"])
            last_flush = time.time()
            while not self._stop or not self.q.empty():
                try:
                    row = self.q.get(timeout=0.1)
                    w.writerow(row)
                except queue.Empty:
                    pass
                if time.time() - last_flush > 0.5:
                    f.flush()
                    last_flush = time.time()

    def log(self, row):
        try:
            self.q.put_nowait(row)
        except queue.Full:
            pass  # descarta si va lleno

    def stop(self):
        self._stop = True
        self.t.join(timeout=1.0)
