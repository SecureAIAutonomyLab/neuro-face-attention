import os
import cv2
import glob
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt

videos_with_cues = glob.glob(r'/media/arundas/data_root/**/**/*.mov')

for i, eachloc in enumerate(videos_with_cues):
    if any(word in eachloc.split('_') for word in ['post.mov', 'pre.mov', 'speech', 'crop.mov']):
        videos_with_cues.pop(i)

for i, eachloc in enumerate(videos_with_cues):
    if 'speech' in eachloc.split('_'):
        videos_with_cues.pop(i)

for i, eachloc in enumerate(videos_with_cues):
    if 'speech' in eachloc.split('_'):
        videos_with_cues.pop(i)

cue_dict = {}

try:
    with open('cue_dict.json') as handle:
        cue_dict = json.load(handle)
    print("** Recovering work from cue_dict.json")
except:
    pass

print(len(videos_with_cues))
for data_loc in videos_with_cues:
    display = False
    dict_key = '/'.join(data_loc.split('/')[-3:])
    # json_loc = 'json_dumps/' + '_'.join(data_loc.split('/')[5:]).split('.')[0] + '.json'
    try:
        if cue_dict[dict_key]["event_id"]==0:
            print("Event ID is zero but is this getting processed?", data_loc)
            pass
        # if the file is already processed and fcount !=0, continue.
        elif cue_dict[dict_key]["event_id"]!=0:
            print("** File already processed. Skipping ", data_loc)
            continue
    except KeyError as error:
        print("[** INFO] Press Enter to skip one frame. \
                          \n[** INFO] Type 0 to skip video. \
                          \n[** INFO] Type 'q' to save and quit.")
        pass
    grab_meta = cv2.VideoCapture(data_loc)
    # Find the frames per second
    fps = int(grab_meta.get(cv2.CAP_PROP_FPS))
    # Read one frame to compute the metadata
    _ret, _frame = grab_meta.read()
    # Check if the video is readable even before starting the process
    if _ret == False:
        print("** Skipping since ret == False")
        cue_dict[dict_key] = {"fcount": -1, "event_id": -1}
        continue
    frame_shape = _frame.shape

    # Apply template Matching
    method = 'cv2.TM_CCOEFF_NORMED'
    method = eval(method)

    # Load the required template based on the video filename
    if data_loc.split('_')[-1].split('.')[0].lower() in ['caw1', 'caw2', 'cw1', 'cw2']:
        img1 = cv2.imread('images/cross.png', 0)
        print("** Processing C-W Scope ", data_loc)
    elif data_loc.split('_')[-1].split('.')[0].lower() in ['wag1', 'wag2', 'wg1', 'wg2']:
        img1 = cv2.imread('images/exclamation.png', 0)
        print("** Processing W-G Scope ", data_loc)
    elif data_loc.split('_')[-2].split('.')[0].lower() in ['caw1', 'caw2', 'cw1', 'cw2']:
        img1 = cv2.imread('images/cross.png', 0)
        print("** Processing C-W Scope ", data_loc)
    elif data_loc.split('_')[-2].split('.')[0].lower() in ['wag1', 'wag2', 'wg1', 'wg2']:
        img1 = cv2.imread('images/exclamation.png', 0)
        print("** Processing W-G Scope ", data_loc)
    else:
        # print("All files are fully processed. Check cue dictionary.")
        print("Didn't load the template. That means ")

    MIN_MATCH_COUNT = 4

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)


    video_capture = cv2.VideoCapture(data_loc)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    fcount = 0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if ret:
            fcount += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left, top, right, bottom = (frame_shape[1]//2, 0, frame_shape[1], frame_shape[0]//2)
            left += 400
            bottom -= 200
            right -= 100
            top += 150

            img2 = frame[top:bottom, left:right]
            kp2, des2 = sift.detectAndCompute(img2,None)
            try:
                matches = flann.knnMatch(des1,des2,k=2)
            except:
                continue
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                try:
                    dst = cv2.perspectiveTransform(pts,M)
                    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                except:
                    matchesMask = None
                display=True
            else:
                #print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                matchesMask = None
                display=False


            cv2.imshow('Video', img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("** Saving final dictionary before skipping video.")
                with open('cue_dict.json', 'w') as handle:
                    json.dump(cue_dict, handle, indent=4)
                break
            # Display the resulting image
            if display:
                text = input("* Which cue? Type 1 or 2, etc. Type 'help' to see options. ")
                try:
                    if text in ('save','SAVE'):
                        print("** Saving final dictionary.")
                        with open('cue_dict.json', 'w') as handle:
                            json.dump(cue_dict, handle, indent=4)
                    if text in ('q','Q'):
                        print("** Saving final dictionary before quitting.")
                        with open('cue_dict.json', 'w') as handle:
                            json.dump(cue_dict, handle, indent=4)
                        quit()
                    elif text in ('help','HELP', 'Help'):
                        print("[** INFO] Press Enter to skip one frame. \
                          \n[** INFO] Type 0 to skip video. \
                          \n[** INFO] Type 'q' to save and quit.")
                    elif int(text) == 0:
                        print("** Chose 0. Skipping ")
                        cue_dict[dict_key] = {"fcount": 0, "event_id": 0}
                        break
                    elif type(int(text)) == int:
                        print("** Frame found using cue: ", fcount)
                        cue_dict[dict_key] = {"fcount": fcount, "event_id": int(text)}
                        break
                    else:
                        pass
                except:
                    continue
        else:
            print("Couldn't find cue in the entire video. Skip and continue.")
            cue_dict[dict_key] = {"fcount": 0, "event_id": 0}
            break
    # # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

print("** Processing done. Saving final dictionary.")
with open('cue_dict.json', 'w') as handle:
    json.dump(cue_dict, handle, indent=4)

# if you want to save as pickle instead of json
# with open('cue_dict.pickle', 'wb') as handle:
#     pickle.dump(cue_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
