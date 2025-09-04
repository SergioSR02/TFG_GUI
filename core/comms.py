import socket, json, time

class UdpPublisher:
    def __init__(self, host="239.0.0.1", port=5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        except Exception:
            pass

    def send(self, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.sock.sendto(data, self.addr)

class UdpSubscriber:
    def __init__(self, host="0.0.0.0", port=5005, timeout=0.01):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(timeout)

    def recv(self):
        try:
            data, _ = self.sock.recvfrom(8192)
            return json.loads(data.decode("utf-8"))
        except socket.timeout:
            return None
