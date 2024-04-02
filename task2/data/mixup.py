import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class MixUp(object):
    def __init__(self, alpha=1.0, sampling_method=1, seed=42):
        self.alpha = alpha
        self.sampling_method = sampling_method
        self.seed = seed
        self.torch_rng = torch.manual_seed(seed)
        self.np_rng = np.random.seed(seed)
        self.use_cuda = torch.cuda.is_available()

    def __call__(self, x, y):
        if self.sampling_method == 1:
            lam = self.get_lambda_beta()
        else:
            lam = self.get_lambda_uniform()

        batch_size = x.size()[0]
        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
            x = x.cuda()
            y = y.cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def get_lambda_beta(self):
        return np.random.beta(self.alpha, self.alpha)

    def get_lambda_uniform(self):
        return np.random.uniform(0, 1)

    def visualize(self, dataloader, num_images=16):
        dataiter = iter(dataloader)
        images, labels = next(dataiter)

        mixed_images = []
        for i in range(num_images):
            mixed_x, _, _, _ = self.__call__(images, labels)
            mixed_images.append(mixed_x[i])

        montage = transforms.Resize((224, 224))(
            torchvision.utils.make_grid(torch.stack(mixed_images), nrow=4)
        )
        montage = montage.permute(1, 2, 0).numpy() * 255
        image = Image.fromarray(montage.astype('uint8'))
        image.save("mixup.png")
        print("Montage of augmented images saved to 'mixup.png'")