import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate

from config import Config


class Utils:

    def custom_collate_fn(batch):
        """
            Collate data of different batch, it is because the boxes and gt_classes have changeable length.
            This function will pad the boxes and gt_classes with zero. (https://github.com/tztztztztz)

            Arguments:
            batch -- list of tuple (im, boxes, gt_classes)

            im_data -- tensor of shape (3, H, W)
            boxes -- tensor of shape (N, 4)
            gt_classes -- tensor of shape (N)
            num_obj -- tensor of shape (1)

            Returns:

            tuple
            1) tensor of shape (batch_size, 3, H, W)
            2) tensor of shape (batch_size, N, 4)
            3) tensor of shape (batch_size, N)
            4) tensor of shape (batch_size, 1)



        # kind of hack, this will break down a list of tuple into
        # individual list
        bsize = len(batch)
        im_data, boxes, gt_classes, num_obj = zip(*batch)
        max_num_obj = max([x.item() for x in num_obj])
        padded_boxes = torch.zeros((bsize, max_num_obj, 4))
        padded_classes = torch.zeros((bsize, max_num_obj,))

        for i in range(bsize):
            padded_boxes[i, :num_obj[i], :] = boxes[i]
            padded_classes[i, :num_obj[i]] = gt_classes[i]

        return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)
        """
        items = list(zip(*batch))
        items[0] = default_collate(items[0])
        items[1] = default_collate(items[1])
        items[2] = list(items[2])
        return items


def augment_img(img, boxes, gt_classes):
    """
    Apply data augmentation. (https://github.com/tztztztztz)
    1. convert color to HSV
    2. adjust hue(.1), saturation(1.5), exposure(1.5)
    3. convert color to RGB
    4. random scale (up to 20%)
    5. translation (up to 20%)
    6. resize to given input size.

    Arguments:
    img -- PIL.Image object
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    im_info -- dictionary {width:, height:}

    Returns:
    au_img -- numpy array of shape (H, W, 3)
    au_boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    au_gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    """

    # img = np.array(img).astype(np.float32)
    boxes = np.copy(boxes).astype(np.float32)

    for i in range(5):
        img_t, boxes_t = random_scale_translation(img.copy(), boxes.copy(), jitter=Config.jitter)
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (boxes_t[:, 1] != boxes_t[:, 3])
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            gt_classes = gt_classes[keep]
            break

    img = random_distort(img, Config.hue, Config.saturation, Config.exposure)
    return img, boxes, gt_classes


def random_scale_translation(img, boxes, jitter=0.2):
    """

    Arguments:
    img -- PIL.Image
    boxes -- numpy array of shape (N, 4) N is number of boxes
    factor -- max scale size
    im_info -- dictionary {width:, height:}

    Returns:
    im_data -- numpy.ndarray
    boxes -- numpy array of shape (N, 4)
    """

    w, h = img.size

    dw = int(w*jitter)
    dh = int(h*jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    # scaled width, scaled height
    sw = w - pl - pr
    sh = h - pt - pb

    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))

    # update boxes accordingly
    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt

    # clamp boxes
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw-1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh-1)

    # if flip
    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw-1) - boxes[:, 2::-2]

    return cropped, boxes


def convert_color(img, source, dest):
    """
    Convert color

    Arguments:
    img -- numpy.ndarray
    source -- str, original color space
    dest -- str, target color space.

    Returns:
    img -- numpy.ndarray
    """

    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort(img, hue=.1, sat=1.5, val=1.5):

    hue = np.random.uniform(-hue, hue)
    sat = rand_scale(sat)
    val = rand_scale(val)

    img = img.convert('HSV')
    cs = list(img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))

    img = img.convert('RGB')
    return img


def random_hue(img, rate=.1):
    """
    adjust hue
    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue
    Returns:
    img -- numpy.ndarray
    """

    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    """
    adjust saturation

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    """
    adjust exposure (In fact, this function change V (HSV))

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img

class WeightLoader(object):
    """ https://github.com/tztztztztz/yolov2.pytorch """
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None

    def load_conv_bn(self, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)

        # make sure the loaded weight is right
        assert size == self.start
