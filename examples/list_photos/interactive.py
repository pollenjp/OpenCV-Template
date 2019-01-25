"""test for interactive"""
import list_photos


def main():
    import os
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
        if re.search(r'.*\.(png|bmp|jpg)', _filename) is not None
    ]

    pprint.pprint(all_files)

    #===================================
    # read images
    images_list = []
    for file_path in all_files:
        im = cv2.imread(file_path)
        print("| file name | {:>15} || shape | {} || dtype | {} |".format(os.path.basename(file_path), im.shape, im.dtype))
        images_list.append(im)

    #===================================
    pickout_index = 0
    im = list_photos.list_photos(images_list=images_list,
                                 canvas_w=1200,
                                 canvas_h=800,
                                 margin_size=50,
                                 canvas_image_ratio=3,
                                 pickout_index=pickout_index,
                                 pickout_offset=[-30,-30])
    im = im.astype(dtype=np.uint8)
    cv2.imshow("Photo List", im)

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
                                     canvas_w=1200,
                                     canvas_h=800,
                                     margin_size=50,
                                     canvas_image_ratio=3,
                                     pickout_index=pickout_index,
                                     pickout_offset=[-30,-30])
        im = im.astype(dtype=np.uint8)
        cv2.imshow("Photo List", im)

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
