import os

class Option:

    def __init__(self):
        self.VIDEO_PATH = os.path.join("C:\\Users\\hamid\\Downloads","intersection.mp4")
        self.MODEL_PATH = os.path.join('.','yolov8n.pt')
        self.TRACKERS = ['bytetrack.yaml','botsort.yaml']
        self.DELEGATE = 0
        self.cls = 2

    def count_region(self,x1,y1,x2,y2):
            region = []
            for y in range(y1,y2):
                for x in range(x1,x2):
                    coordinates = (x, y)
                    region.append(list(coordinates))

            return region
