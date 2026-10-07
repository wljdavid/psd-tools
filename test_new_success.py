import time
import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage


def show_ndarray(img, title='Image', scale_factor=1):
    # 缩放图像
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective_transform(img, src_points, target_points, size):
    # 计算输入图像在输出图像中的偏移量，使其居中
    # offset_x = (size[0] - img.shape[1]) // 2
    # offset_y = (size[1] - img.shape[0]) // 2

    # 构建平移矩阵
    # translation_matrix = np.array([[1, 0, 0],
    #                                [0, 1, 0],
    #                                [0, 0, 1]])

    # 透明度掩膜
    alpha_channel = img[:, :, 3]  # 获取图像的 alpha 通道
    mask = (alpha_channel == 0).astype(np.uint8)  # 创建透明度掩膜
    img[mask] = [0, 0, 0, 0]
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, target_points)

    '''cv '''
    size = (img.shape[1],img.shape[0])
    ret_data = cv2.warpPerspective(img, perspective_matrix, size, flags=cv2.INTER_NEAREST)

    return ret_data


def smart_test3(psd_path: str, img_path: str):
    psd: PSDImage = PSDImage.open(psd_path)

    '''cv'''
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
        w, h = mask_size
        l, r = mask_offsite
        psd_image = psd.composite()

        # 透视变换
        src_points = np.array([
            [0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]
        ], dtype=np.float32)
        target_points = np.array([
            trans[0:2], trans[2:4], trans[4:6], trans[6:8]
        ], dtype=np.float32)
        # img = cv2.resize(img, mask_size)
        # img = cv2.resize(img, (598, 764))
        img = cv2.resize(img, psd_image.size)
        result = perspective_transform(img, src_points, target_points, (psd.width, psd.height))
        # result = np.rot90(result, k=1)
        # result = np.flipud(result)
        # result = perspective_transform(img, src_points, target_points, (psd.width, psd.height))
        # show_ndarray(result, scale_factor=0.5)
        # 贝塞尔变换
        vector_paths = []
        for path in vm_paths:
            path_data = [(point.anchor[0] * psd_image.size[0], point.anchor[1] * psd_image.size[1]) for point in
                         path._items]
            vector_paths.append(path_data)
        paths = [
            np.array(vector_paths, dtype=np.int32)
        ]

        # 创建掩膜
        height, width, _ = result.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, pts=paths, color=255)

        # 创建透明度通道
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)
        # 将 result 中透明部分的 alpha 通道设为 0
        alpha_channel[result[..., 3] == 0] = 0

        # 创建带透明通道的图像
        result = cv2.merge((result[..., 2], result[..., 1], result[..., 0], alpha_channel))

        # 将掩膜应用到图像上
        result = cv2.bitwise_and(result, result, mask=mask)

        # 将结果转换为PIL图像，并进行旋转和翻转
        out_image = Image.fromarray(result)
        out_image = out_image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

        # 假设 psd_image 是一个PIL图像对象
        psd_image.paste(out_image, (0, 0), mask=out_image)

        # 保存结果图像
        ret_path = "./{}_{}.png".format("out", int(time.time()))
        psd_image.save(ret_path)
    pass


if __name__ == '__main__':
    # files = ["./test/V领6.psd"]
    files = ["./test/2.psd"]
    img_path = "./test/upload.png"
    for item in files:
        smart_test3(item, img_path)
        time.sleep(0.1)
