import time

import cv2
import numpy as np
from PIL import Image

from psd_tools import PSDImage


def perspective_transform(image, src_points, target_points,size):
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, target_points)
    print(perspective_matrix)


    '''cv '''
    #image = cv2.resize(image,size)
    #ret_data = cv2.warpPerspective(image, perspective_matrix,size,flags=cv2.INTER_LINEAR)
    # image = Image.fromarray(ret_data)
    '''pil'''
    perspective_matrix = perspective_matrix.reshape(9,1)
    image = image.transform(size, Image.PERSPECTIVE, tuple(perspective_matrix[:8]),Image.NEAREST)

    return image


def smart_test3(psd_path: str, img_path: str):
    psd = PSDImage.open(psd_path)
    '''pil'''
    # img = Image.open(img_path)
    '''cv'''
    img = cv2.imread(img_path)
    trans: list = []
    mask_size: tuple = ()
    mask_offsite: list = []
    try:
        layers = psd.layser()
        for layer in layers:
            if layer.name == "qianpian" and layer.kind == "smartobject":
                smart = layer.smart_object
                config = smart.config
                trans = config['Trnf']
                mask = layer.mask
                mask_size = mask.size
                mask_offsite = [mask.left, mask.top]
                vector_mask = layer.vector_mask
                vm_bbox = vector_mask.bbox
                vm_paths = vector_mask.paths

    except Exception as e:
        print(e)
        pass
    if len(trans) != 0:
        vector_paths = []
        for path in vm_paths:
            path_data = [(point.anchor[0] * 1000, point.anchor[1] * 1000) for point in path._items]
            vector_paths.append(path_data)
        paths = [
            np.array(vector_paths, dtype=np.int32)
        ]

        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, pts=paths, color=255)

        result = cv2.bitwise_and(img, img, mask=mask)

        ret_path = "./{}_{}.png".format("out", int(time.time()))
        cv2.imwrite(ret_path, result)


if __name__ == '__main__':
    files = ["./test/2.psd"]
    img_path = "./test/red.png"
    for item in files:
        smart_test3(item,img_path)
        time.sleep(0.1)