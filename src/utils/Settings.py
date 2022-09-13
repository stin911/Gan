import json
import os


def set_upsetting():
    # Data to be written
    dictionary = {
        # Number of workers for dataloader
        "workers": 2,
        # Batch size during training
        "batch_size": 128,
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        "image_size": 64,
        # Number of channels in the training images. For color images this is 3
        "nc": 3,
        # Size of z latent vector (i.e. size of generator input)
        "nz": 100,
        # Size of feature maps in generator
        "ngf": 64,
        # Size of feature maps in discriminator
        "ndf": 64,
        # Number of training epochs
        "num_epochs": 100,
        # Learning rate for optimizers
        "lr": 0.0001,
        # Beta1 hyperparam for Adam optimizers
        "beta1": 0.5,
        # Number of GPUs available. Use 0 for CPU mode.
        "ngpu": 1
    }
    #

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    with open("C:/Users/alexn/PycharmProjects/DisneyCharacter/Data/Setting/sett.json", "w") as outfile:
        outfile.write(json_object)


def load_setting(src):
    f = open(src)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    workers = data["workers"]
    # Batch size during training
    batch_size = data["batch_size"]
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = data["image_size"]
    # Number of channels in the training images. For color images this is 3
    nc = data["nc"]
    # Size of z latent vector (i.e. size of generator input)
    nz = data["nz"]
    # Size of feature maps in generator
    ngf = data["ngf"]
    # Size of feature maps in discriminator
    ndf = data["ndf"]
    # Number of training epochs
    num_epochs = data["num_epochs"]
    # Learning rate for optimizers
    lr = data["lr"]
    # Beta1 hyperparam for Adam optimizers
    beta1 = data["beta1"]
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = data["ngpu"]
    return workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu
