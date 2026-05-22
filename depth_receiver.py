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
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape((msg.height, msg.width))

        with self.lock:
            self.depth = depth

    def get_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()