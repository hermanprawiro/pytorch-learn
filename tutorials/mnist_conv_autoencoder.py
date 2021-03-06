import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_path = './mnist_conv_autoencoder'

if not os.path.exists(img_path):
    os.mkdir(img_path)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1), # (4, 14, 14)
        #     nn.ReLU(True),
        #     nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1), # (16, 7, 7)
        #     # nn.ReLU(True),
        #     # nn.Conv2d(16, 64, kernel_size=3, stride=2) # (64, 2, 2)
        # )
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2),
        #     # nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh()
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), # 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) # 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), # 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), # 1, 28, 28
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run(batch_size, max_epochs, lr, log_interval):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(MNIST(download=False, root="./data", transform=data_transform), batch_size=batch_size, shuffle=True)

    model = Autoencoder()
    if device == 'cuda':
        model.cuda()
        torch.backends.cudnn.benchmark = True
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(max_epochs):
        for img, _ in train_loader:
            img = img.to(device)
            # img = img.view(img.shape[0], -1)
            
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('Epoch [{}/{}], loss = {:.4f}'.format(epoch + 1, max_epochs, loss.item()))

        if epoch % log_interval == 0:
            pic = output.detach().to('cpu')
            pic = 0.5 * (pic + 1)
            pic = pic.clamp(0, 1)
            # pic = pic.view(pic.shape[0], 1, 28, 28)
            save_image(pic, os.path.join(img_path, 'image_{}.png'.format(epoch)))
    torch.save(model.state_dict(), os.path.join(img_path, 'model.pth'))

if __name__ == "__main__":
    num_epochs = 150
    batch_size = 512
    learning_rate = 1e-3
    run(batch_size, num_epochs, learning_rate, 10)