""" enclose in rectangle """
import numpy as np
import cv2

def enclose_in_rectangle(image_arr,
                         coordinate_leftup,
                         coordinate_rightdown,
                         rectangle_thickness=4,
                         rectangle_color=(0, 255, 0),  # BGR
                         margin_color=(255,255,255)):
    """
    | image_arr            | numpy.array | a numpy.array image, int array, shape(H, W)
    | coordinate_leftup    | list,tuple  | [x,y], horizantal axis x, vertical axis y
    | coordinate_rightdown | list,tuple  | [x,y], horizantal axis x, vertical axis y
    | rectangle_thickness  | int         | an odd number, rectangle frame thickness
    | rectangle_color      | tuple       | rectangle frame thickness
    | margin_color         | tuple       | 0-255, RGB=(margin_color,margin_color,margin_color)
    Return
    | canvas               | numpy.array | np.uint8
    """
    #===================================
    # Canvasを用意
    canvas_h, canvas_w = image_arr.shape[0] + 2*rectangle_thickness, image_arr.shape[1] + 2*rectangle_thickness
    print(type(canvas_h))
    canvas = np.zeros(shape=(canvas_h, canvas_w, 3), dtype=np.uint8)
    for i in range(3):
        canvas[:,:,i] = margin_color[i]

    #===================================
    # 写真を配置
    canvas[rectangle_thickness:-rectangle_thickness, rectangle_thickness:-rectangle_thickness, :] = image_arr

    #===================================
    # 長方形を描く
    #canvas = cv2.rectangle(img=canvas, pt1=coordinate_leftup, pt2=coordinate_rightdown, color=rectangle_color, thickness=rectangle_thickness)
    # cv2.rectangle で指定するthicknessは不安定な（単純にpixel数ではない）ので以下のように自分で描画した
    leftup_x   , leftup_y    = np.array(coordinate_leftup)    + np.array([rectangle_thickness, rectangle_thickness])
    rightdown_x, rightdown_y = np.array(coordinate_rightdown) + np.array([rectangle_thickness, rectangle_thickness])
    color                    = np.array(rectangle_color)
    canvas[leftup_y-rectangle_thickness : leftup_y, leftup_x - rectangle_thickness : rightdown_x+1+rectangle_thickness, :]           = color  # 上辺
    canvas[rightdown_y+1 : rightdown_y+1+rectangle_thickness, leftup_x - rectangle_thickness : rightdown_x+1+rectangle_thickness, :] = color  # 底辺
    canvas[leftup_y-rectangle_thickness : rightdown_y+1+rectangle_thickness, leftup_x - rectangle_thickness : leftup_x, :]           = color  # 左辺
    canvas[leftup_y-rectangle_thickness : rightdown_y+1+rectangle_thickness, rightdown_x+1 : rightdown_x+1+rectangle_thickness, :]   = color  # 右辺

    return canvas


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    ################################################################################################################################################################################################################################################
    #parser.add_argument('path_data_csv',       action='store', nargs=None, const=None, default=None, type=str, choices=None, required=True,  help='Directory path where your taken photo files are located.')
    parser.add_argument('--img_path',           action='store', nargs=None, const=None, default=None, type=str, choices=None, required=True,  help='image directory path')
    parser.add_argument('--rectangle_thickness',action='store', nargs='?',  const=4,    default=4,    type=int, choices=None, required=False, help='')
    args = parser.parse_args()

    #===================================
    # 画像読み込み
    im = cv2.imread(filename=args.img_path)

    #===================================
    canvas = enclose_in_rectangle(image_arr=im,
                                  coordinate_leftup=[0,0],
                                  coordinate_rightdown=[40, 40],
                                  rectangle_thickness=args.rectangle_thickness)
    print("im.min() : {}".format(im.min()))
    print("type(im) : {}".format(type(im)))
    print("im.shape : {}".format(im.shape))

    #===================================
    # cv2.imshow
    # https://docs.opencv.org/3.0-beta/modules/highgui/doc/user_interface.html?highlight=cv2.imshow#cv2.imshow
    im = canvas.astype(dtype=np.uint8)
    cv2.namedWindow("Sample Image")
    cv2.imshow("Sample Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
