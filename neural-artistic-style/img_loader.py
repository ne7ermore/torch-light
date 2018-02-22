import torch
from torchvision import transforms

from PIL import Image

try:
    import matplotlib.pyplot as plt

    def imshow(tensor, imsize=512, title=None):
        image = tensor.clone().cpu()
        image = image.view(*tensor.size())
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(5)
except:
    plt = None
    imshow = None
    print("Device do not support matplotlib")

class IMG_Processer(object):
    def __init__(self, img_size=800, path="images/"):
        self.img_path = path
        self.img_size = img_size

    def toTensor(self, img):
        encode = transforms.Compose([transforms.Resize(self.img_size),
               transforms.ToTensor(),
               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
               transforms.Lambda(lambda x: x.mul_(255)),
            ])

        return encode(Image.open(img))

    def img2tensor(self, style_img_name, content_img_name):
        _style, _content = map(self.toTensor,
            list(map(lambda n: self.img_path+n, [style_img_name, content_img_name])))

        return _style, _content

    def tensor2img(self, tensor, epoch):
        decode = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
               transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                    std=[1,1,1]),
               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
               ])
        tensor = decode(tensor)

        loader = transforms.Compose([transforms.ToPILImage()])
        img = loader(tensor.clamp_(0, 1))

        img.save(self.img_path + "/result_{}.jpg".format(epoch))

if __name__ == '__main__':
    if imshow is not None:
        ip = IMG_Processer()
        _style, _content = ip.img2tensor('vangogh_starry_night.jpg', 'nm_logo.jpg')

        plt.ion()

        plt.figure()
        imshow(_style)

        plt.figure()
        imshow(_content)

    else:
        print("Do not support")