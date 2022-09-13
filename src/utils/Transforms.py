import torch
from torchvision.transforms import ToTensor
import cv2


class ResizeImage:
    """Resize Image"""

    def __call__(self, image):
        dim = (64, 64)
        reimgsized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
        return reimgsized

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):


            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            sample = sample.transpose((2, 0, 1))
            sample = torch.from_numpy(sample).float()

            sample = sample.unsqueeze(0)
            return sample


