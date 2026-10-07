import time
import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

def perspective_transform(image, src_points, target_points, size):
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, target_points)
    print(perspective_matrix)

    '''cv'''
    image = cv2.resize(image, size)
    ret_data = cv2.warpPerspective(image, perspective_matrix, size, flags=cv2.INTER_LINEAR)
    image = Image.fromarray(ret_data)

    return image

def smart_test3(psd_path: str, img_path: str):
    psd: PSDImage = PSDImage.open(psd_path)
    img = cv2.imread(img_path)
    trans = []
    mask_size = ()
    mask_offsite = []

    try:
        layers = psd.layser()
        if "qianpian" in [layer.name for layer in layers] and any(layer.kind == "smartobject" for layer in layers):
            # Process layers only if the required layer names and kinds exist
            for layer in layers:
                if layer.name == "qianpian" and layer.kind == "smartobject":
                    smart = layer.smart_object
                    config = smart.config
                    trans = config['Trnf']
                    mask = layer.mask
                    mask_size = mask.size
                    mask_offsite = [mask.left, mask.top]
                    vector_mask = layer.vector_mask
                    vm_paths = vector_mask.paths

            if len(trans) != 0:
                img = cv2.resize(img, psd.size)
                w, h = mask_size
                l, r = mask_offsite
                psd_image = psd.composite()

                # 矩阵变换
                src_points = np.array([
                    trans[0:2], trans[2:4], trans[4:6], trans[6:8]
                ], dtype=np.float32)
                target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                trans_img = perspective_transform(img, src_points, target, (1000,1000))

                # 贝塞尔变换
                vector_paths = []
                for path in vm_paths:
                    path_data = [(point.anchor[0] * 1000, point.anchor[1] * 1000) for point in path._items]
                    vector_paths.append(path_data)
                paths = [
                    np.array(vector_paths, dtype=np.int32)
                ]

                # 创建掩膜
                width, height = trans_img.size
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, pts=paths, color=255)

                # 创建透明度通道
                alpha_channel = np.full((height, width), 255, dtype=np.uint8)

                # Create a 4-channel alpha image (RGBA) with alpha_channel data
                alpha_data = np.array(alpha_channel, dtype=np.uint8)
                alpha_img = Image.fromarray(alpha_data, 'L').convert("RGBA")

                # Merge alpha channel and trans_img
                result = Image.alpha_composite(trans_img.convert("RGBA"), alpha_img)
                result = np.array(result)

                # 将掩膜应用到图像上
                result = cv2.bitwise_and(result, result, mask=mask)

                # 将结果转换为PIL图像，并进行旋转和翻转
                out_image = Image.fromarray(result)
                out_image = out_image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

                # 假设 psd_image 是一个PIL图像对象
                psd_image.paste(out_image, (l, r), mask=out_image)

                # 保存结果图像
                ret_path = "./{}_{}.png".format("out", int(time.time()))
                psd_image.save(ret_path)

    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    # files = ["./test/V领6.psd"]
    files = ["./test/2.psd"]
    img_path = "./test/test.png"
    for item in files:
        smart_test3(item, img_path)
        time.sleep(0.1)
