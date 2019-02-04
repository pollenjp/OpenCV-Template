""" list photos """
import copy
import numpy as np
import cv2

def list_photos(images_list,
                canvas_w=1200,
                canvas_h=800,
                margin_size=[50, 25],
                canvas_image_ratio=3,
                pickout_index=0,
                pickout_offset=[0, 0]):
    """
    | images_list    | list | list of numpy.array image
    |                |      | [ np.array([...]), ..., np.array([...])
    |                |      |   np.array([...]), ..., np.array([...]) ]
    | margin_size    | int  | canvasの内側の余白
    | pickout_index  | int  | pickout_index枚目の写真を強調するため
    | pickout_offset | list | pickout_index枚目の写真の中心座標をずらすオフセット
    """
    assert margin_size[0] > pickout_offset[0]
    assert margin_size[1] > pickout_offset[1]
    #===================================
    # Canvasを用意
    canvas = np.full(shape=(canvas_h, canvas_w, 3), fill_value=255)

    #===================================
    # 写真の配置の開始位置を終わり位置を指定
    start_coordinate = np.array([canvas_h - margin_size[1], margin_size[0]])  # 左下
    end_coordinate   = np.array([margin_size[1], canvas_w - margin_size[0]])  # 右上

    #===================================
    # canvas内部の一枚あたりの写真の最大幅(縦横)を指定
    # 縦横比率はcanvasに合わせ,canvasの1/3とする
    image_w, image_h = int(canvas_w / canvas_image_ratio), int(canvas_h / canvas_image_ratio)

    #===================================
    # 写真の枚数
    images_num = len(images_list)
    # 中心座標が配置される間隔
    images_center_interval_w = int((end_coordinate[1] - start_coordinate[1] - image_w) / (images_num - 1))
    images_center_interval_h = int((start_coordinate[0] - end_coordinate[0] - image_h) / (images_num - 1))

    #===================================
    center_coordinates = []
    center1 = start_coordinate + np.array([-1, 1]) * np.array([image_h, image_w], dtype=np.int) // 2 + (1-1) * np.array([-1, 1]) * np.array([images_center_interval_h, images_center_interval_w], dtype=np.int)
    center_coordinates.append(center1)
    for i in range( 1,images_num):
        center = center1 + i * np.array([-1, 1]) * np.array([images_center_interval_h, images_center_interval_w], dtype=np.int)
        center_coordinates.append(center)

    #===================================
    # pickout_index枚目の写真を強調
    center_coordinates[pickout_index] += np.array(pickout_offset[::-1], dtype=np.int)

    #===================================
    # image_w, image_h に収まるようにリサイズする
    def resize_image(image, frame_width, frame_height):
        """
        width, height の中に収まり,かつ比率を変えないようにリサイズする.
        """
        im = copy.deepcopy(image)
        im_h, im_w = im.shape[0], im.shape[1]

        new_w = int(im_w * min(frame_width/im_w, frame_height/im_h))
        new_h = int(im_h * min(frame_width/im_w, frame_height/im_h))

        resized_image = cv2.resize(im, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
        return resized_image

    resized_images_list = [resize_image(image=image, frame_width=image_w, frame_height=image_h) for image in images_list]

    #===================================
    for (y,x), re_im in zip(center_coordinates[::-1], resized_images_list[::-1]):
        y_width, x_width, _ = re_im.shape

        _y_min = y-int(y_width/2)
        if y_width % 2 == 0:
            _y_max = y+int(y_width/2)
        else:
            _y_max = y+int(y_width/2) + 1

        _x_min = x-int(x_width/2)
        if x_width % 2 == 0:
            _x_max = x+int(x_width/2)
        else:
            _x_max = x+int(x_width/2) + 1

        canvas[_y_min:_y_max, _x_min:_x_max, :] = re_im

    return canvas


if __name__ == "__main__":
    import os
    import sys
    import pathlib
    import re
    import pprint
    import matplotlib.pyplot as plt

    #===================================
    print(sys.version)
    print("np         : {}".format(np.__version__))
    print("cv2        : {}".format(cv2.__version__))

    #===================================
    HOME_Path = pathlib.Path(os.getcwd()).parents[1]
    img_Path  = HOME_Path / "img"
    print("path name | exist | path\n" + 
          "========================")
    print("HOME_Path | {:5} | {}".format(HOME_Path.exists(), str(HOME_Path)))
    print("img_Path  | {:5} | {}".format(img_Path.exists(),  str(img_Path)))

    #===================================
    # filename list
    all_files = [
        os.path.join(os.path.abspath(_dirpath), _filename)
        for _dirpath, _dirnames, _filenames in os.walk(str(img_Path))
        for _filename in _filenames
        if re.search(r'^(?!.*^\.).*\.(png|bmp|jpg|tif)', _filename) is not None
    ]  # .(ピリオド)で始まるファイルを無視

    pprint.pprint(all_files)

    #===================================
    # read images
    images_list = []
    for file_path in all_files:
        im = cv2.imread(file_path)
        print("| file name | {:>15} || shape | {} || dtype | {} |".format(os.path.basename(file_path), im.shape, im.dtype))
        images_list.append(im)

    #===================================
    canvas = list_photos(images_list=images_list,
                         canvas_w=1200,
                         canvas_h=800,
                         margin_size=[50, 50],
                         canvas_image_ratio=3)
    im = canvas.astype(dtype=np.int8)
    print("im.min() : {}".format(im.min()))
    print("type(im) : {}".format(type(im)))
    print("im.shape : {}".format(im.shape))

    #===================================
    # Matplotlib
    print("matplot")
    print("canvas.dtype : {}".format(canvas.dtype))
    im = cv2.cvtColor(canvas.astype(dtype=np.uint8),
                      cv2.COLOR_BGR2RGB)
    print("im.min() : {}".format(im.min()))
    fig = plt.figure()
    nrows, ncols, idx = 1, 1, 0
    idx += 1
    ax = fig.add_subplot(nrows, ncols, idx)

    ax.imshow(im)

    ax.set_title(label="im.shape:{}".format(im.shape))
    ax.set_xlabel(xlabel="im[1]")
    ax.set_ylabel(ylabel="im[0]")
    plt.imsave("./matplot_out.png", im)

    #===================================
    # cv2.imshow
    # https://docs.opencv.org/3.0-beta/modules/highgui/doc/user_interface.html?highlight=cv2.imshow#cv2.imshow
    im = canvas.astype(dtype=np.uint8)
    cv2.namedWindow("Photo List")
    cv2.imshow("Photo List", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
