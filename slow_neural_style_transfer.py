import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

## ---------- CLASSES ---------- ##

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
    
## ----------------------------- ##


## --------- FUNCTIONS --------- ##

def load_image(image_name, device, loader):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def create_loader(imsize):
    return transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])

def compute_losses(content_features, style_features, generated_features, alpha, beta):
    content_loss = 0
    style_loss = 0

    for generated_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
        content_loss += torch.mean((generated_feature - content_feature) ** 2)

        # Gram Matrix
        G = generated_feature.view(generated_feature.shape[1], -1).mm(
            generated_feature.view(generated_feature.shape[1], -1).t()
        )
        A = style_feature.view(style_feature.shape[1], -1).mm(
            style_feature.view(style_feature.shape[1], -1).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * content_loss + beta * style_loss
    return total_loss, content_loss, style_loss

def train(content_img, style_img, generated_img, model, optimizer, epochs, alpha, beta, callback):
    
    with tqdm(range(epochs)) as tq:
        for step in tq:
            
            generated_features = model(generated_img)
            content_features = model(content_img)
            style_features = model(style_img)

            total_loss, content_loss, style_loss = compute_losses(content_features, style_features, generated_features, alpha, beta)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step%50==0:
                callback(
                    int(((step+1)/epochs)*100), 
                    f"Time Left : {int((tq.total - tq.n) / tq.format_dict['rate'] if tq.format_dict['rate'] and tq.total else 0)} s",
                    f"Time Taken : {int(tq.format_dict['elapsed'])} s",
                    f"Total Loss : {int(total_loss.item())}, Content Loss : {int(content_loss.item())}, Style Loss : {int(style_loss.item())}"
                )
        
    return generated_img

## ----------------------------- ##


## --------- EXPORT --------- ##

def slow_neural_style_transfer(content_img_upload, style_img_upload, epochs, learning_rate, alpha, beta, callback):
    
    imsize = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG().to(device).eval()
    loader = create_loader(imsize)
    inverse_loader = transforms.ToPILImage()
    
    content_img = load_image(content_img_upload, device, loader)
    style_img = load_image(style_img_upload, device, loader)
    generated_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=learning_rate)
    
    generated_img = train(content_img, style_img, generated_img, model, optimizer, epochs, alpha, beta, callback)
    returnable_img = inverse_loader(torch.squeeze(generated_img, 0))
    returnable_img.filename = f"generated_{int(time.time())}.png"
    return returnable_img
    # return generated_img
    
## ----------------------------- ##


## ------------ TEST ------------ ##

# epochs = 2000
# learning_rate = 0.001
# alpha = 10
# beta = 0.01

# generated = slow_neural_style_transfer(Image.open("./tests/slow/input/content.png"), Image.open("./tests/slow/input/style.png"), epochs=epochs, learning_rate=learning_rate, alpha=alpha, beta=beta)
# save_image(generated, "./tests/slow/output/generated.png")

## ------------------------------ ##