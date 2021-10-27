import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target = self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() # discard previous(unprocessed) frame
                except queue.Empty:
                    pass
                self.q.put(frame)

    def read(self):
        return self.q.get()

cap = VideoCapture(0)
print('check')
while True:
    time.sleep(.5) # simulate time between events
    print('working')
    frame = cap.read()
    print('got frame')
    cv2.imshow("frame", frame)
    print('show frame')
    if chr(cv2.waitKey(1) & 255) == 'q':
        break