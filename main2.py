from ultralytics import YOLO
import supervision as sv
import os
import cv2
from options import Option

#⏬⏬⏬

#⚠️ TWO IMPORTANT note: 1-FIRSTLY THIS MODEL IS RUNNING ON GPU (YOU CAN CHANGE IT FROM < OPTIONS/Option/DELEGATE='cpu' )⚠️

#⚠️ 2- if you run this script on GPU , you can plaily see that video starts so fast before cars appear and further
#  calculations start. this difference and latency 
# stems from switching between cpu and gpu since we are running detection model on gpu and some operations with opencv,...
# alse note that some funcitons are costly for GPU and may be running here. ⚠️

#⚠️ 3-there are some weak detections. they are because of using lite model <yolov8n.py>. I chose that since I aimed it to
#  run on cpu to be versatile on all devices⚠️
# 
# 4- Try to use vscode terminal(not the vscode run icon) while running these scripts. ( >> python main2.py) 

op = Option()

class Detector:
    def __init__(self):
        self.ST=sv.Point(0,220)
        self.EN=sv.Point(640,220)
        self.ST2=sv.Point(500,0)
        self.EN2=sv.Point(500,360)
        self.rec_x1 = 50
        self.rec_x2 = 590
        self.rec_y1 = 100
        self.rec_y2 = 200
        

    def main(self):
        model = YOLO(op.MODEL_PATH)

        line_zone = sv.LineZone(self.ST,self.EN)
        line_spec = sv.LineZoneAnnotator(thickness=1,color=sv.Color.red(),
                                         text_thickness=1,text_color=sv.Color.black(),
                                         text_scale=0.5,text_offset=1.5,text_padding=10,
                                         custom_in_text='Up',custom_out_text="Down")
        
        line_zone_right = sv.LineZone(self.ST2,self.EN2)
        line_spec_right = sv.LineZoneAnnotator(thickness=1,color=sv.Color.red(),
                                         text_thickness=1,text_color=sv.Color.black(),
                                         text_scale=0.5,text_offset=1.5,text_padding=10,
                                         custom_in_text='right',custom_out_text="left")

        box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=1,
                                        text_scale=0.5,text_color=sv.Color.blue(),
                                        color=sv.Color.white())

        for result in model.track(source=op.VIDEO_PATH,tracker=op.TRACKERS[0],
                                  device=op.DELEGATE,persist=True,
                                  stream=True,
                                  agnostic_nms=True):
            frame = result.orig_img

            detections = sv.Detections.from_ultralytics(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            detections = detections[detections.class_id == op.cls]
            labels = [f"#{tracker_id} {model.model.names[class_id]}" 
                      for _,_,conf,class_id,tracker_id, in detections]
            
            frame = box_annotator.annotate(scene=frame,detections=detections,
                                           labels=labels)
            line_zone.trigger(detections=detections)
            line_spec.annotate(frame=frame,line_counter=line_zone)
            line_zone_right.trigger(detections=detections)
            line_spec_right.annotate(frame=frame,line_counter=line_zone_right)

            cv2.rectangle(frame, (self.rec_x1, self.rec_y1), (self.rec_x2,self.rec_y2), (0,255,0), 1)
#⚠️   
            counter=0
            lst = []
            lst2 = []
            if detections.xyxy is not None:
                for coordinate in detections.xyxy:

                    x1,y1,x2,y2 = coordinate
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    center = (x1+((x2-x1)/2),y1+(y2-y1)/2)
                    lst.append(center)
                    if detections.tracker_id is not None:
                        lst2.append(detections.tracker_id[counter])
                    counter+=1

            
            lst3 = []

            for index in range(len(lst)):
                if list(lst[index]) in op.count_region(x1=self.rec_x1,y1=self.rec_y1,x2=self.rec_x2,y2=self.rec_y2) and len(lst)==len(lst2):
                    lst3.append(lst2[index])



            print(f"these cars are at the center of the intersection {[i for i in lst3]}")


            
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

Detector().main()



