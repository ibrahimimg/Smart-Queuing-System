import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''
    
    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        ### Initialize any class variables desired ###
        # To be updated
        self.w = 0.0
        self.h = 0.0
        self.current_frame = None
        self.pr_frame = None
        self.net = None
        self.exec_net = None
        # Initialize the inference ingine
        self.ie = IECore()
        
        try:
            #Load the Intermediate Representation model
            #self.net = IENetwork(self.model_structure, self.model_weights)
            self.net = self.ie.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
            
        # Get the input layer
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        '''
        # Load the model network
        self.exec_net = self.ie.load_network(network=self.net,device_name=self.device,num_requests=1)
    
    def async_req_get(self, input_dict):
        # Start an asynchronous request ###
        self.exec_net.start_async(request_id=0,inputs=input_dict)
        # Wait for the request to be complete.
        status = self.exec_net.requests[0].wait(-1)
        if status==0:
            # Extract and return the output results
            result = self.exec_net.requests[0].outputs[self.output_name]
            return result
   
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        self.current_frame = image
        # process the current frame
        self.pr_frame = self.preprocess_input(self.current_frame)
        input_dict={self.input_name: self.pr_frame}
        # get output results
        result = self.async_req_get(input_dict)
        coords, output_image = self.preprocess_outputs(result)
        return coords,output_image
    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        '''
        p1 = (coords[0],coords[1])
        p2 = (coords[2],coords[3])
        # Draw bounding boxes onto the image
        cv2.rectangle(image, p1, p2, (0, 255, 0) , 2)
        
        return

    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''
        coordinates=list()
        #for box in outputs[0][0]:
        for b in range (len(outputs[0][0])):
            box = outputs[0][0][b]
            confidence = box[2]
            if confidence > self.threshold:
                x_min,x_max = map(lambda b : int(b*self.w), [box[3],box[5]])
                y_min,y_max = map(lambda b : int(b*self.h), [box[4],box[6]])
                coordinates.append([x_min,y_min,x_max,y_max])
                coords = [x_min,y_min,x_max,y_max]
                self.draw_outputs(coords, self.current_frame)
                
        return coordinates, self.current_frame

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        # Pre-process the image as needed #
        _width=self.input_shape[3]
        _height=self.input_shape[2]
        p_image = cv2.resize(image, (_width, _height))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(self.input_shape[0], self.input_shape[1], _height, _width)
        
        return p_image


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pd.w = initial_w
    pd.h = initial_h
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h))
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        out_video.release()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print (fps)
        print (total_inference_time)
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--device', default='CPU', type=str)
    parser.add_argument('--video', default=None, type=str)
    parser.add_argument('--queue_param', default=None, type=str)
    parser.add_argument('--output_path', default='/results', type=str)
    parser.add_argument('--max_people', default=2, type=int)
    parser.add_argument('--threshold', default=0.60, type=float)
    
    args=parser.parse_args()

    main(args)
