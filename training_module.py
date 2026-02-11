import pytorch_lightning as L
from torch import optim

# define the LightningModule
class TrainingModule(L.LightningModule):
    def __init__(self, model,loss,metric):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metric = metric

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred,y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        metric_value = self.metric(y_pred,y)
        self.log("validation_metric", metric_value,prog_bar=True)
        return metric_value

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    


# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader,TensorDataset
# import torchvision.datasets as datasets


# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# # lets extract the actual datapoints and annotations so that we can explore them
# x_train = mnist_trainset.data.numpy()
# y_train = mnist_trainset.targets.numpy()
# x_test = mnist_testset.data.numpy()
# y_test = mnist_testset.targets.numpy()
# x_train_normalized = x_train / 255.
# x_test_normalized = x_test / 255.

# # in the next step, we also need to reshape our input to fit our input layer later on. 
# # This is due to pytorch expecting a definition for how many channels your input sample has, as we 
# # deal with gray scale this is 1.
# x_train_normalized= x_train_normalized.reshape(-1, 28, 28, 1)
# x_test_normalized = x_test_normalized.reshape(-1, 28, 28, 1)

# # To work with PyTorch we also need to convert our numpy arrays to tensors.
# x_train_normalized = torch.from_numpy(x_train_normalized).float()
# x_test_normalized = torch.from_numpy(x_test_normalized).float()
# y_train = torch.from_numpy(y_train)
# y_test = torch.from_numpy(y_test)

# marvin = torch.nn.Sequential(
#   torch.nn.Flatten(),
#   torch.nn.Linear(in_features=784,out_features=128),
#   torch.nn.ReLU(),
#   torch.nn.Dropout(0.2),
#   torch.nn.Linear(in_features=128,out_features=10)
# )

# train_dataset = TensorDataset(x_train_normalized, y_train)
# train_dl = DataLoader(train_dataset, batch_size=32,shuffle=True)

# test_dataset = TensorDataset(x_test_normalized, y_test)
# test_dl = DataLoader(test_dataset, batch_size=32,shuffle=False)

# def accuracy(y_pred,y):
#     y_pred_argmax = torch.argmax(y_pred,dim=1)
#     accuracy = torch.sum(y_pred_argmax == y)/len(y)
#     return accuracy

# loss_fn = torch.nn.CrossEntropyLoss()
# training_mod = TrainingModule(
#     model=marvin,
#     loss=loss_fn,
#     metric=accuracy
# )


# trainer = pl.Trainer(
#     max_epochs=20
# )
# trainer.fit(model=training_mod, train_dataloader=train_dl,val_dataloaders=test_dl)
