import os
import numpy as np
import cv2
#from glob import glob
import glob

#_IMAGE_SIZE = 224


def cal_for_frames(video_path):
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    #frames = os.listdir(video_path)
    frames.sort()
    #print (frames)

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow



def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{:04d}.jpg".format(i)),
                    flow[:, :, 0])
        if not os.path.exists(os.path.join(flow_path.format('u'))):
            os.makedirs(os.path.join(flow_path.format('u')))
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:04d}.jpg".format(i)),
                    flow[:, :, 1])
        if not os.path.exists(os.path.join(flow_path.format('v'))):
            os.makedirs(os.path.join(flow_path.format('v')))



def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return






if __name__ == '__main__':

    folders = ['./flow/']

    for folder in folders:
        train_or_val = glob.glob(folder + '*')
        #print train_or_val


        for sub_folder in train_or_val:
            class_files = glob.glob(sub_folder + '/*')
            #print class_files

            for sen_folders in class_files:
                class_index = glob.glob(sen_folders + '/*')
                #print class_index

                for class_frames in class_index:
                    #vid_file = glob.glob(vid_class)
                    #print class_frames

                    parts = class_frames.split('/')
                    train_or_val = parts[2]  #train_or_val（train）
                    class_files = parts[3]   #class_files（1peace）
                    class_frames= parts[4]   #class_index（1peace_0003）

                    print(train_or_val)
                    print( class_files )
                    print (class_frames)
                    # train_or_test = parts[2]
                    # print (train_or_test)

                    video_paths = '  ' + train_or_val + '/' + class_files + '/' + class_frames + '/'
                    #
                    flow_paths = '  ' + train_or_val + '/' + class_files + '/' + class_frames + '/'
                    #
                    extract_flow(video_paths, flow_paths)