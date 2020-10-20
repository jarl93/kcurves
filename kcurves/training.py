# libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
def train_synthetic(model, data_set, num_epochs, lambda_, beta_, regularization, lr=1e-4, verbose=False):
    """
    Add documentation!
    """
    print("Starting training on synthetic data...")

    # Use Adam optmizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    data_set.set_mode(mode='train')
    list_loss = []
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data_batch in enumerate(data_set.get_batch()):

            # get the data and labels from the generator get_batch
            x, y = data_batch

            # convert the numpy arrays into Pytorch tensors
            x = torch.Tensor(x)
            y = torch.Tensor(y)

            # print("x shape: ", x.shape)
            # print("y shape: ", y.shape)

            x = x.to(device)

            # Resize the input accordingly
            x = x.view(-1, model.input_dim)
            optimizer.zero_grad()

            # Get the reconstrunction from the autoencoder
            x_reconstructed = model(x)

            h = model.encoder(x)

            # Compute the loss of the batch
            loss = loss_function(x, x_reconstructed, h, model, lambda_, beta_, regularization)

            list_loss.append(loss)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if verbose:
                if batch_idx % 50 == 0:
                    print(datetime.datetime.now(), end = '\t')
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * data_set.batch_size, data_set.N,
                               100. * batch_idx / data_set.num_batches, loss.item() / data_set.batch_size))

        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / data_set.N))

    print("Training DONE!")
    plt.plot(list_loss)
    plt.title("Loss training")
    plt.show()
    return None


def train_mnist(num_epochs):
    """
    Add documentation!
    """
    model.train()
    print("Starting training on MNIST dataset...")
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)

            # Resize the input accordingly
            x = x.view(-1, input_dim)
            optimizer.zero_grad()

            # Get the reconstrunction from the autoencoder
            x_reconstructed = model(x)

            # Compute the loss of the batch
            loss = loss_function(x, x_reconstructed, None, model, lambda_=1e-3, beta_=0, regularization='L1')

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(datetime.datetime.now(), end = '\t')
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(y), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(y)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    print("Training DONE!")

    return None