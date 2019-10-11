import os
import cv2
import glob
import json
import numpy as np
from matplotlib import pyplot as plt

data_root = '/media/arundas/data_root/'

with open('cue_dict.json') as handle:
    cue_dict = json.load(handle)

# for each_video_fname, each_video_details in other_video:
for each_video_fname, each_video_details in cue_dict.items():

    subject, study, fname = each_video_fname.split('/')
    paradigm = fname.split('_')[1].split('.')[0]
    data_loc = data_root + each_video_fname
    out_dir = os.path.join(data_root,'segments',subject,study,paradigm.lower())
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    try:
        grab_meta = cv2.VideoCapture(data_loc)
        # Find the frames per second
        fps = int(grab_meta.get(cv2.CAP_PROP_FPS))
        # Read one frame to compute the metadata
        _, _frame = grab_meta.read()
        frame_shape = _frame.shape
        _min_side = min(frame_shape[:2])
        # Release handle to the webcam
        grab_meta.release()
        cv2.destroyAllWindows()
    except:
        print("File {} is not working".format(data_loc))
        pass

    video_capture = cv2.VideoCapture(data_loc)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')


    fcount = each_video_details['fcount']
    event_id = each_video_details['event_id']

    if paradigm.lower() in ['caw1', 'caw2', 'cw1', 'cw2']:
        epoch_start = int(fcount/event_id)
        if fcount%3==0:
            epoch_end = epoch_start + int(np.ceil(7.4*fps))
        else:
            epoch_end = epoch_start + int(7.4*fps)
    elif paradigm.lower() in ['wag1', 'wag2', 'wg1', 'wg2']:
        if fcount%3==0:
            epoch_start = fcount - int(np.ceil(1.502*fps)-((event_id-1)*fps*7.5))
            epoch_end = epoch_start + int(np.ceil(7.4*fps))
        else:
            epoch_start = fcount - int((1.502*fps)-(event_id-1)*fps*7.5)
            epoch_end = epoch_start + int(7.4*fps)

    print("####### STUDY DETAILS #######\n")
    print('Subject: {}, Study: {}, File Name: {}, Paradigm: {}'.format(subject, study, fname, paradigm))
    print("\n####### VIDEO DETAILS #######\n")
    print("Input Video Resolution: {}".format(str(_min_side)+"p"))
    print("Total Number of Frames: {}".format(total_frames))
    print("Fcount, EventID: ", fcount, event_id)
    print("First Epoch Start: ", epoch_start)
    print("First Epoch End: ", epoch_end)
    print("FPS: ", fps)
    
    count = 0 # frame id or fcount of the while loop.
    segcount = 1
    
    out_fname = out_dir + "/" + str(fname.split('.')[0]) + '_segment_' + str(segcount) + '.avi'
    fid = cv2.VideoWriter(out_fname, fourcc, fps, (frame_shape[1], frame_shape[0]))
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if ret:
            count+=1
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Display the resulting image
            # cv2.imshow('Video', frame)
            
            if count>=epoch_start and count<=epoch_end:
                fid.write(frame)
            
            if count==epoch_end:                
                print("Extracted Segment #", segcount)
                
                segcount+=1
                if segcount==51:
                    print("Extracted 50 epochs. Halt.\n")
                    break

                # Update the output segment filename and the video writer 
                out_fname = out_dir + "/" + str(fname.split('.')[0]) + '_segment_' + str(segcount) + '.avi'
                fid = cv2.VideoWriter(out_fname, fourcc, fps, (frame_shape[1], frame_shape[0]))        
            
                if paradigm.lower() in ['caw1', 'caw2', 'cw1', 'cw2']:
                    if fcount%3==0:
                        epoch_start = epoch_end
                        epoch_end = epoch_start + int(np.ceil(7.72*fps))
                    else:
                        epoch_start = epoch_end
                        epoch_end = epoch_start + int(7.72*fps)
                elif paradigm.lower() in ['wag1', 'wag2', 'wg1', 'wg2']:
                    if fcount%3==0:
                        epoch_start = epoch_end
                        epoch_end = epoch_start + int(np.ceil(7.66*fps))
                    else:
                        epoch_start = epoch_end
                        epoch_end = epoch_start + int(7.66*fps)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

print("Done!")
# Release handle to the webcam
video_capture.release()
fid.release()
cv2.destroyAllWindows()
