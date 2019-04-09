import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import matplotlib.pyplot as plt
import time
import numpy as np
import cv2


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        if img == 'q':
            break
        # img = 'data/dog.jpg'
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            # plt.imshow(r_image)
            # break
    yolo.close_session()

#
# ############读取显示视频
# cap = cv2.VideoCapture('data/1.mp4')  # 打开相机
# while (True):
#     ret, frame = cap.read()  # 捕获一帧图像
#     if ret:
#         cv2.imshow('frame', frame)
#         cv2.waitKey(30)
#     else:
#         break
# cap.release()  # 关闭相机
# cv2.destroyAllWindows()  # 关闭窗口



def my_detect_video(yolo, video_path, output_path=""):
    import cv2

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    return_value, frame = vid.read()

    while True:
        return_value, frame = vid.read()
        if return_value:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#Opencv转PIL
            image = yolo.detect_image(image)
            result = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)#显示的时候再PIL转回Opencv
            #
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            #
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            # plt.imshow(result)
            # if isOutput:
            #     out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    yolo.close_session()


####检测视频
my_detect_video(YOLO(), input("Input video filename:\n"))

#########检测图片
# detect_img(YOLO())







#
# FLAGS = None
#
#
# # class YOLO defines the default value, so suppress any default here
# parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
# '''
# Command line options
# '''
# parser.add_argument(
#     '--model', type=str,
#     help='path to model weight file, default ' + YOLO.get_defaults("model_path")
# )
#
# parser.add_argument(
#     '--anchors', type=str,
#     help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
# )
#
# parser.add_argument(
#     '--classes', type=str,
#     help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
# )
#
# parser.add_argument(
#     '--gpu_num', type=int,
#     help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
# )
#
# parser.add_argument(
#     '--image', default=False, action="store_true",
#     help='Image detection mode, will ignore all positional arguments'
# )
# '''
# Command line positional arguments -- for video detection mode
# '''
# parser.add_argument(
#     "--input", nargs='?', type=str,required=False,default='./path2your_video',
#     help = "Video input path"
# )
#
# parser.add_argument(
#     "--output", nargs='?', type=str, default="",
#     help = "[Optional] Video output path"
# )
#
# FLAGS = parser.parse_args()
#
#
# if FLAGS.image:
#     """
#     Image detection mode, disregard any remaining command line arguments
#     """
#     print("Image detection mode")
#     if "input" in FLAGS:
#         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
#     detect_img(YOLO(**vars(FLAGS)))
# elif "input" in FLAGS:
#     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
# else:
#     print("Must specify at least video_input_path.  See usage with --help.")
