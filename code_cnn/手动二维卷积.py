'''
手动实现二维卷积操作
'''
import numpy as np

# 卷积： 假设 Stride = 1, Padding = 0, img 和 kernel 都是 np.ndarray.
def conv2d(img, kernel):
    # 输入大小
    height, width, in_channels = img.shape
    # 卷积核大小
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    # 计算输出大小
    out_height, out_width = height - kernel_height + 1, width - kernel_width + 1
    # 输出特征图
    feature_maps = np.zeros(shape=(out_height, out_width, out_channels))
    # 对于每一个卷积核
    for oc in range(out_channels):              # Iterate out_channels (# of kernels)
        for h in range(out_height):             # Iterate out_height
            for w in range(out_width):          # Iterate out_width
                for ic in range(in_channels):   # Iterate in_channels
                    patch = img[h: h + kernel_height, w: w + kernel_width, ic]
                    feature_maps[h, w, oc] += np.sum(patch * kernel[:, :, ic, oc])

    return feature_maps