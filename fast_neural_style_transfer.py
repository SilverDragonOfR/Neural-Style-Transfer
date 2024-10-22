import time
import torch
from PIL import Image
from collections import namedtuple
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets

import warnings
warnings.filterwarnings("ignore")

## ---------- CLASSES ---------- ##

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*vgg_pretrained_features[:4])
        self.slice2 = nn.Sequential(*vgg_pretrained_features[4:9])
        self.slice3 = nn.Sequential(*vgg_pretrained_features[9:16])
        self.slice4 = nn.Sequential(*vgg_pretrained_features[16:23])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)

## ----------------------------- ##


## --------- FUNCTIONS --------- ##

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def gram_matrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (c * h * w)

def create_transforms(imsize):
    return transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_checkpoint_model(style_name, check_point):
    return torch.load(f"./fnst_pretrained_styles/models/{style_name}/{style_name}_{check_point}.pth")

def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def deprocess(image_tensor):
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    return image_np.transpose(1, 2, 0)

def save_sample(style_name, transformer, image_samples, batches_done, device):
    transformer.eval()
    with torch.no_grad():
        output = transformer(image_samples.to(device))
    image_grid = denormalize(torch.cat((image_samples.cuda(), output.cuda()), 2))
    save_image(image_grid, f"./fnst_pretrained_styles/samples/{style_name}/{style_name}_{batches_done}_sample.png", nrow=4)
    transformer.train()

def train(style_image, dataset_path, batch_size, learning_rate, epochs, alpha, beta, checkpoint_interval):

    imsize = 256

    style_name = os.path.basename(style_image).split(".")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = datasets.ImageFolder(dataset_path, create_transforms(imsize))
    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    optimizer = Adam(transformer.parameters(), learning_rate)
    l2_loss = nn.MSELoss().to(device)

    style = create_transforms(imsize)(Image.open(style_image)).repeat(batch_size, 1, 1, 1).to(device)
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    image_samples = []
    for path in random.sample(glob.glob(f"{dataset_path}/*/*.jpg"), 8):
        image_samples.append(create_transforms(imsize)(Image.open(path)))
    image_samples = torch.stack(image_samples)

    for epoch in range(epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = transformer(images_original)

            features_original = vgg(images_original)
            features_transformed = vgg(images_transformed)

            content_loss = alpha * l2_loss(features_transformed.relu2_2, features_original.relu2_2)

            style_loss = sum(l2_loss(gram_matrix(ft_y), gm_s[: images.size(0), :, :]) for ft_y, gm_s in zip(features_transformed, gram_style)) * beta

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics["content"].append(content_loss.item())
            epoch_metrics["style"].append(style_loss.item())
            epoch_metrics["total"].append(total_loss.item())

            # Log metrics
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    epochs,
                    batch_i + 1,
                    len(train_dataset) // batch_size,
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            batches_done = epoch * len(dataloader) + batch_i + 1
            
            if checkpoint_interval > 0 and batches_done % checkpoint_interval == 0:
                os.makedirs(f"./fnst_pretrained_styles/samples/{style_name}/", exist_ok=True)
                os.makedirs(f"./fnst_pretrained_styles/models/{style_name}/", exist_ok=True)
                save_sample(style_name, transformer, image_samples, batches_done, device)
                torch.save(transformer.state_dict(), f"./fnst_pretrained_styles/models/{style_name}/{style_name}_{batches_done}.pth")
                
def test(image, checkpoint_model, device):
    
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(checkpoint_model)
    transformer.eval()

    image_tensor = Variable(image).to(device)

    with torch.no_grad():
        generated_image = denormalize(transformer(image_tensor)).cuda()

    return generated_image

## ----------------------------- ##


## --------- EXPORT --------- ## 

def fast_neural_style_transfer(content_img_upload, style_name, check_point):
    
    imsize = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = create_transforms(imsize=imsize)
    inverse_transform = transforms.ToPILImage()
    content_img = transform(Image.open(content_img_upload)).unsqueeze(0).to(device)
    checkpoint_model = get_checkpoint_model(style_name, check_point)
    
    generated_img = test(content_img, checkpoint_model, device)
    returnable_img = inverse_transform(torch.squeeze(generated_img, 0))
    returnable_img.filename = f"generated_{int(time.time())}.png"
    time.sleep(3)
    return returnable_img
    # return generated_img
    
## ----------------------------- ##


## ----------- TRAIN ----------- ##

# batch_size = 1
# learning_rate = 0.001
# epochs = 1
# alpha = 1e5
# beta = 1e10
# check_point_interval = 500

# train("./fnst_pretrained_styles/styles/Starry Nights.png", "dataset_fnst", batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, alpha=alpha, beta=beta, checkpoint_interval=check_point_interval)

## ----------------------------- ##


## ------------ TEST ------------ ##

# check_point = 1500

# generated = fast_neural_style_transfer(Image.open("./tests/fast/input/content.png"), "Starry Nights", check_point=check_point)
# save_image(generated, "./tests/fast/output/generated.png")

## ------------------------------ ##