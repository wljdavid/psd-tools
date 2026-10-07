import time

import cv2
import numpy as np
from PIL import Image

from psd_tools import PSDImage


def perspective_transform(img, src_points, target_points, size):
    # 透明度掩膜
    try:
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, target_points)
        '''cv '''
        ret_data = cv2.warpPerspective(img, perspective_matrix, size, flags=cv2.INTER_NEAREST)
    except Exception as e:
        print(e)
        ret_data = img

    return ret_data


def pastImage(psd:Image,info:any,trans:[],mask_size:[],vm_paths:[]):
    img = cv2.imread(info['img_path'], cv2.IMREAD_UNCHANGED)
    try:
        # 透视变换
        src_points = np.array([
            [0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]
        ], dtype=np.float32)
        target_points = np.array([
            trans[0:2], trans[2:4], trans[4:6], trans[6:8]
        ], dtype=np.float32)

        # 旋转和翻转
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
        result = perspective_transform(img, src_points, target_points, (psd.width, psd.height))

        # 将结果转换为PIL图像
        out_image = Image.fromarray(result)

        psd.paste(out_image, (0, 0), mask=out_image)
    except Exception as e:
        print(e)
    return psd


def smart_test3(psd_path: str, infos: any):
    psd: PSDImage = PSDImage.open(psd_path)
    psd_image = psd.composite()
    try:
        layers = psd.layser()
        for layer in layers:
            items = [i for i in infos if i['layer_name'] == layer.name]
            if len(items) == 1 and layer.kind == "smartobject":
                smart = layer.smart_object
                config = smart.config
                trans = config['Trnf']
                mask = layer.mask
                mask_size = mask.size
                mask_offsite = [mask.left,mask.top]
                vector_mask = layer.vector_mask
                vm_bbox = vector_mask.bbox
                vm_paths = vector_mask.paths
                if len(trans) != 0:
                    psd_image = pastImage(psd_image,items[0],trans,mask_size,vm_paths)
        # 保存结果图像
        ret_path = "./{}_{}.jpg".format("out", int(time.time()))
        psd_image.save(ret_path)
        return ret_path
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':

    # files = ["./test/V领6.psd"]
    files = ["./test/chimal.psd"]
    # img_path = "./test/upload.png"
    for item in files:
        smart_test3(item,[
            {'layer_name': 'zuoxiu','img_path': './output_image.png'},
            {'layer_name': 'qianpian', 'img_path': './output_image.png'},
            {'layer_name': 'houpian', 'img_path': './output_image.png'}
        ])
        time.sleep(0.1)