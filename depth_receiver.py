from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import threading

class DepthReceiver:
    def __init__(self, topic):
        self.node = Node()
        self.depth = None
        self.lock = threading.Lock()

        # ✅ FIXED LINE
        self.node.subscribe(Image, topic, self.callback)

    def callback(self, msg: Image):
        if not self.lock.acquire(blocking=False):
            return  # drop frame — main thread holds lock, prevents gz queue backup
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32)
            self.depth = depth.reshape((msg.height, msg.width))
        finally:
            self.lock.release()

    def get_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()