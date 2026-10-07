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
    image = cv2.resize(image,size)
    ret_data = cv2.warpPerspective(image, perspective_matrix,size,flags=cv2.INTER_LINEAR)
    image = Image.fromarray(ret_data)
    '''pil'''
    # perspective_matrix = perspective_matrix.reshape(9,1)
    # image = image.transform(size, Image.PERSPECTIVE, tuple(perspective_matrix[:8]),Image.NEAREST)

    return image


def smart_test3(psd_path:str,img_path:str):
    psd: PSDImage = PSDImage.open(psd_path)
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
                mask_offsite = [mask.left,mask.top]
                vector_mask = layer.vector_mask
                vm_bbox = vector_mask.bbox
                vm_paths = vector_mask.paths

    except Exception as e:
        print(e)
        pass
    if len(trans) != 0:
        # img = img.resize(mask_size)
        w,h = mask_size
        l,r = mask_offsite
        psd_image = psd.composite()
        src_points = np.array([
            trans[0:2], trans[2:4], trans[4:6],trans[6:8]
        ], dtype=np.float32)
        target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        out_image = perspective_transform(img,src_points,target,mask_size)
        '''pil'''
        # out_image = out_image.rotate(0, Image.BICUBIC)
        # out_image =img.transform(mask_size,Image.QUAD,trans)

        # 贝塞尔变换
        vector_paths = []
        for path in vm_paths:
            path_data = [(point.anchor[0] * 600, point.anchor[1] * 600) for point in path._items]
            vector_paths.append(path_data)
        paths = [
            np.array(vector_paths, dtype=np.int32)
        ]

        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, pts=paths, color=255)

        result = cv2.bitwise_and(img, img, mask=mask)
        result = np.dstack((result, np.full(result.shape[:2], 255, dtype=np.uint8)))

        out_image = Image.fromarray(result)
        out_image = out_image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

        psd_image.paste(out_image,(l+30,r+100),mask=out_image)
        ret_path = "./{}_{}.png".format("out", int(time.time()))
        psd_image.save(ret_path)
    # img
    pass


if __name__ == '__main__':
    # files = ["./test/V领6.psd"]
    files = ["./test/2.psd"]
    img_path = "./test/test.png"
    for item in files:
        smart_test3(item,img_path)
        time.sleep(0.1)