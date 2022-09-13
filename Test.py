import torch
from src.Generator import Generator
import numpy as np
from src.utils.utility import show_image
import cv2
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

nz = 100
netG = Generator(ngpu).to(device)
checkpoint = torch.load("./Data/saves/love95.pt")
netG.load_state_dict(checkpoint['netG'])
t = torch.randn(128, nz, 1, 1, device=device)
netG.zero_grad()
fake = netG(t)
for data in fake:
    data = data.permute(1, 2, 0)
    a = ((data.cpu().detach().numpy()))
    print(np.shape(a))
    a = cv2.resize(a, (600,400), interpolation=cv2.INTER_CUBIC)
    show_image(a)

