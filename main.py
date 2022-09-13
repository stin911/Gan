import os
from matplotlib import pyplot as plt

import torch.optim as optim
from src.Generator import Generator, Discriminator
import torch.nn as nn
from src.utils.utility import weights_init
import torchvision.utils as vutils

from src.DataLoader import CustomImageDataset
import torch
from torchvision import transforms
from src.utils.Transforms import ResizeImage, ToTensor
from src.utils.utility import show_image
from src.utils.Settings import load_setting
import warnings
from torchsummary import summary
warnings.filterwarnings("ignore")
workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu = \
    load_setting("C:/Users/alexn/PycharmProjects/DisneyCharacter/Data/Setting/sett.json")

if __name__ == "__main__":

    dt = CustomImageDataset(annotations_file=".//Data//train_part.csv", img_dir=".//Data//IMG//",
                            transform=transforms.Compose(
                                [ResizeImage(), ToTensor()]))

    dataloader = torch.utils.data.DataLoader(dt, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    load = True
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # to load pretrained
    if load:
        print("Loading")
        checkpoint = torch.load("./Data/saves/love95.pt")
        netD.load_state_dict(checkpoint['netD'])
        netG.load_state_dict(checkpoint['netG'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
    else:
        netD.apply(weights_init)
        netG.apply(weights_init)
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Print the model
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    criterion = nn.BCELoss()

    print("Starting Training Loop...")
    errD_real = 0
    errD_fake = 0
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        print("Epoch " + str(epoch))
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'netD': netD.state_dict(),
                'netG': netG.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'errD_real': errD_real,
                'errD_fake': errD_fake,
            }, "./Data/saves/love"+str(epoch)+".pt")
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device, dtype=torch.float)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    # plot loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

