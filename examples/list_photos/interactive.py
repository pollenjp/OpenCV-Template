"""test for interactive"""
import list_photos


def main(img_path,
         canvas_w=1200,
         canvas_h=800,
         margin_size=[50, 50],
         canvas_image_ratio=3,
         pickout_offset=[-30,-30],
         image_list_step=2):
    """
    | img_path | str | image directory path
    """
    import sys
    import pathlib
    import re
    import pprint
    import numpy as np
    import cv2

    #===================================
    # print config
    print(sys.version)
    print("np         : {}".format(np.__version__))
    print("cv2        : {}".format(cv2.__version__))

    #===================================
    # filename list
    image_path_list = [
        os.path.join(os.path.abspath(_dirpath), _filename)
        for _dirpath, _dirnames, _filenames in os.walk(str(img_path))
        for _filename in _filenames
        if re.search(r'.*\.(png|bmp|jpg|tif)', _filename) is not None
    ]

    #===================================
    # 間引き調整
    images_list = image_path_list[::image_list_step]

    pprint.pprint(image_path_list)

    #===================================
    # read images
    images_list = []
    for file_path in image_path_list:
        im = cv2.imread(file_path)
        print("| file name | {:>15} || shape | {} || dtype | {} |".format(os.path.basename(file_path), im.shape, im.dtype))
        images_list.append(im)

    #===================================
    pickout_index = 0
    im = list_photos.list_photos(images_list=images_list,
                                 canvas_w=canvas_w,
                                 canvas_h=canvas_h,
                                 margin_size=margin_size,
                                 canvas_image_ratio=canvas_image_ratio,
                                 pickout_index=pickout_index,
                                 pickout_offset=pickout_offset)
    im = im.astype(dtype=np.uint8)
    cv2.imshow("Photo List", im)
    cv2.moveWindow('Photo List', 20, 20)

    while True:
        # is there a waitkey table? - OpenCV Q&A Forum
        # http://answers.opencv.org/question/100740/is-there-a-waitkey-table/?answer=100745#post-id-100745
        key_code = cv2.waitKey(0) & 0xFF 
        print("| key_code            | {}".format(key_code))
        print("| type(key_code)      | {}".format(type(key_code)))
        #if True:
        #    continue
        if key_code == 27 or key_code == 113:           # ESCキー : 終了
            break
        elif key_code == 102:                           # fキー : 次の写真 (forward)
            if pickout_index < len(images_list) - 1:
                # 最後の写真を選んでいなければindexを1足す
                pickout_index += 1
        elif key_code == 98:                            # bキー : 前の写真 (backward)
            if pickout_index > 0:
                # 最初の写真を選んでいなければindexを1引く
                pickout_index -= 1

        #===================================
        # cv2.imshow
        # https://docs.opencv.org/3.0-beta/modules/highgui/doc/user_interface.html?highlight=cv2.imshow#cv2.imshow
        im = list_photos.list_photos(images_list=images_list,
                                     canvas_w=canvas_w,
                                     canvas_h=canvas_h,
                                     margin_size=margin_size,
                                     canvas_image_ratio=canvas_image_ratio,
                                     pickout_index=pickout_index,
                                     pickout_offset=pickout_offset)
        im = im.astype(dtype=np.uint8)
        cv2.imshow("Photo List", im)

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    ################################################################################################################################################################################################################################################
    #parser.add_argument('path_data_csv',      action='store', nargs=None, const=None, default=None, type=str, choices=None, required=True,  help='Directory path where your taken photo files are located.')
    parser.add_argument('--img_dir_path',      action='store', nargs=None, const=None, default=None, type=str, choices=None, required=True,  help='image directory path')
    parser.add_argument('--margin_size_x',     action='store', nargs='?',  const=5,    default=5,    type=int, choices=None, required=False, help='')
    parser.add_argument('--margin_size_y',     action='store', nargs='?',  const=10,   default=10,   type=int, choices=None, required=False, help='')
    parser.add_argument('--canvas_image_ratio',action='store', nargs='?',  const=4,    default=4,    type=int, choices=None, required=False, help='')
    parser.add_argument('--pickout_offset_x',  action='store', nargs='?',  const=0,    default=0,    type=int, choices=None, required=False, help='')
    parser.add_argument('--pickout_offset_y',  action='store', nargs='?',  const=-25,  default=-25,  type=int, choices=None, required=False, help='')
    args = parser.parse_args()

    assert os.path.exists(args.img_dir_path)

    main(img_path=args.img_dir_path,
         canvas_w=600,
         canvas_h=400,
         margin_size=[args.margin_size_x, args.margin_size_x],
         canvas_image_ratio=args.canvas_image_ratio,
         pickout_offset=[args.pickout_offset_x, args.pickout_offset_y])
