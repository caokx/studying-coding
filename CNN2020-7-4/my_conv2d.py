import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

img_path = "../imgs/vmgirl.jpg"
img = cv2.imread(img_path)
img_size = img.shape

X = torch.tensor(img).view(img_size[2], img_size[1], img_size[0])


class con2d(nn.Module):
    def __init__(self, channels=3):
        super(con2d, self).__init__()
        self.channels = channels
        kernel = [[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel/9
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=1, groups=self.channels)
        return x
















input_x = cv2.imread(img_path)
cv2.namedWindow("1", 0);
cv2.resizeWindow("1", 290, 435);

cv2.imshow("1", input_x)
input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1)
gaussian_conv = con2d()
out_x = gaussian_conv(input_x)
out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
cv2.namedWindow("2", 0);
cv2.resizeWindow("2", 290, 435);
cv2.imshow("2", out_x)
cv2.waitKey(0)

# X = X.view(img_size[0], img_size[1], img_size[2])
# img = X.numpy()
# print(img.shape)


# cv2.namedWindow("1", 0);
# cv2.resizeWindow("1", 290, 435);
# cv2.imshow("1", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.destroyWindow("1")
