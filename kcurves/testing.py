# libraries
import torch
import numpy as np

def test_mnist(data_loader, show_images=True, images_to_show=10):
    print("Starting testing on MNIST dataset...")

    model.eval()

    X_2D = None  # array to store the latent vectors
    labels = None  # labels of the test data

    for batch_idx, data in enumerate(data_loader):

        x, y = data

        # save the original image for later comparison
        img_original = x[0].squeeze()

        # Send the data to the device
        x = x.to(device)

        # Resize the input accordingly
        x = x.view(-1, input_dim)

        # Encode the data to see how the result looks
        h = model.encoder(x).detach().numpy()

        if batch_idx == 0:
            X_2D = h
            labels = y
        else:
            X_2D = np.vstack((X_2D, h))
            labels = np.hstack((labels, y))

        # Get the reconstrunction from the autoencoder
        x_reconstructed = model(x)

        if show_images and batch_idx < images_to_show:
            # resize the tensor to see the image
            img_reconstructed = x_reconstructed.view(-1, 28, 28).detach().numpy()[0]
            print ("Batch: {}".format(batch_idx))
            # plot the original image and the reconstructed image
            imshow(img_original, img_reconstructed)

    print("Testing DONE!")

    return X_2D, labels


def test_synthetic(model, data_set, mode_forced):
    """
    Add documentation!
    """
    model.eval()
    data_set.set_mode(mode=mode_forced)
    X_2D = None
    for batch_idx, data in enumerate(data_set.get_batch()):

        # print ("batch_idx: ", batch_idx)

        # get the data and labels from the generator get_batch
        x, y = data

        # convert the numpy arrays into Pytorch tensors
        x = torch.Tensor(x)
        y = torch.Tensor(y)

        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape)

        # Send the data to the device
        x = x.to(device)

        # Resize the input accordingly
        x = x.view(-1, model.input_dim)

        # Encode the data to see how the result looks
        h = model.encoder(x).detach().numpy()

        if batch_idx == 0:
            X_2D = h
        else:
            X_2D = np.vstack((X_2D, h))

        # Get the reconstrunction from the autoencoder
        x_reconstructed = model(x)

    return X_2D
