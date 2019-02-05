"""
apply_mask
@pollenjp - github / polleninjp@gmail.com
"""

import numpy as np
import cv2

def apply_mask(image_arr,
               colored_mask_arr,
               is_pred_mask_arr,
               alpha,
               beta,
               gamma,
               ):
    """
    masked_arr = image_arr * alpha + mask_arr * beta + gamma
    > cv2.addWeighted()
    > The function can be replaced with a matrix expression:
    > dst = src1*alpha + src2*beta + gamma;
    > [Operations on Arrays — OpenCV 2.4.13.7 documentation](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#addweighted)

    | image_arr            | numpy.array | a numpy.array image, uint8 array
    | colored_mask_arr     | numpy.array | 色を塗るためのmask
    | is_pred_mask_arr     | numpy.array | 予測された場所かどうかの２値で埋まったmask (predicted:1, not predicted:0)
    | alpha                | float       |
    | beta                 | float       |
    | gamma                | float       |
    Return
    | masked_arr           | numpy.array | np.uint8
    """
    assert image_arr.shape == colored_mask_arr.shape, "image_arr.shape: {}, colored_mask_arr.shape: {}".format(image_arr.shape, colored_mask_arr.shape)
    no_masked_arr = image_arr * (1-is_pred_mask_arr)
    masked_arr = (image_arr * alpha + colored_mask_arr * beta + gamma).astype(dtype=np.uint8) * is_pred_mask_arr
    masked_arr = masked_arr + no_masked_arr
    assert masked_arr.dtype == np.uint8
    return masked_arr

def create_colored_mask(colored_mask_arr,
                        is_pred_mask_arr,
                        pred_result_dict_list,
                        rectangle_size=(28, 28),
                        ):
    """
    | colored_mask_arr     | numpy.array | 色を塗るためのmask
    | pred_result_dict_list| list        | {"coordinate": (addr_x, addr_y),
    |                      |             |  "state": state,
    |                      |             |  "score": score}
    |                      |             | addr_x | int   |   | 元画像における 28x28pixel 短形(左上)のx座標//rectangle_size[0]
    |                      |             | addr_y | int   |   | 元画像における 28x28pixel 短形(左上)のy座標//rectangle_size[1]
    |                      |             | state  | int   | 0 | 癌でない
    |                      |             |        | int   | 1 | 癌である
    |                      |             | score  | float |   | stateである確率(のようなもの)
    | rectangle_size       | tuple       | tuple(x_size, y_size)             短形サイズ
    Return
    | colored_mask_arr     | numpy.array | 色を塗るためのmask
    | is_pred_mask_arr     | numpy.array | 予測された場所かどうかの２値で埋まったmask (predicted:1, not predicted:0)
    """
    for pred_result_dict in pred_result_dict_list:
        _x, _y = pred_result_dict["coordinate"]
        _x_size, _y_size = rectangle_size
        colored_mask_arr[_y:_y+_y_size, _x:_x+_x_size, :] = \
            create_colored_mask_block(pred_state=pred_result_dict["state"],
                                      pred_score=pred_result_dict["score"],
                                      rectangle_size=rectangle_size)
        is_pred_mask_arr[_y:_y+_y_size, _x:_x+_x_size, :] = 1
    assert colored_mask_arr.dtype == np.uint8
    return colored_mask_arr, is_pred_mask_arr

def create_colored_mask_block(pred_state,
                              pred_score,
                              rectangle_size=(28,28),
                              ):
    """
    | pred_state           | int         | 0 : 癌でない
    |                      | int         | 1 : 癌である
    | pred_score           | float       | stateである確率(のようなもの)
    | rectangle_size       | tuple       | tuple(x_size, y_size)             短形サイズ
    """
    mask_arr = np.full(shape=(rectangle_size[0], rectangle_size[1], 3), fill_value=0, dtype=np.uint8)
    if pred_state:  # pred_state によって配色を変える
        _color = (0, int(255*(1-pred_score)), int(255*pred_score))  # BGR
    else:
        _color = (0, int(255*pred_score), int(255*(1-pred_score)))
    mask_arr = cv2.rectangle(img=mask_arr,
                             pt1=(0, 0),
                             pt2=rectangle_size,
                             color=_color,
                             thickness=-1)  # draw a filled rectangle
    assert mask_arr.dtype == np.uint8
    return mask_arr

    #src = image_BG[addr_y*size:(addr_y+1)*size, addr_x*size:(addr_x+1)*size]
    #brightness = np.mean(src)
    #if brightness > 170:
    #    br = 1
    #else:
    #    br = brightness/170
    #alpha, beta, gamma = 1-br, br, 0
    #mask_arr = src1*alpha + src2*beta + gamma
    #mask_arr = cv2.addWeighted(src1=src, alpha=1-br, src2=mask_arr, beta=br, gamma=0, dtype=np.uint8)
    # > The function can be replaced with a matrix expression:
    # > dst = src1*alpha + src2*beta + gamma;
    # > [Operations on Arrays — OpenCV 2.4.13.7 documentation](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#addweighted)
