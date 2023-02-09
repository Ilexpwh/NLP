import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from flickerdata import FlickerData
from torch.utils.data import DataLoader
from Params import Params
from model import ImageCaptionGenerator
from model import train_ICG, predict_one, performance_scores
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

if __name__ == "__main__":

    train_model = True
    reset_model = False

    # Load in preprocessed dataset
    if os.path.exists("Datasets/dataset.p"):
        train_dataset: FlickerData
        dev_dataset: FlickerData
        test_dataset: FlickerData
        with open("Datasets/dataset.p", "rb") as f:
            train_dataset, dev_dataset, test_dataset = pickle.load(f)
    else:
        raise FileExistsError("Run flickerdata.py first to create dataset")

    vocab_lc = train_dataset.vocab_lc # vocabulary from train data
    p = Params(vocab_size=len(vocab_lc)) # parameters

    # Dataloaders
    train_dataloader = DataLoader(dataset= train_dataset, batch_size= p.batch_size, shuffle= True)
    val_dataloader = DataLoader(dataset= dev_dataset, batch_size= p.batch_size, shuffle= False)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size= p.batch_size, shuffle= False)

    # Model instance
    model = ImageCaptionGenerator(vocab_lc, p)
    model.to(p.device)
    if (os.path.exists("Models/icg.pt")) and (not reset_model):
        model.load_state_dict(torch.load("Models/icg.pt"))

    # Loss and Optimizer
    PAD = vocab_lc["<PAD>"]
    loss_func = nn.CrossEntropyLoss(ignore_index= PAD, reduce= "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr= p.lr)

    if train_model:
        train_history = train_ICG(model, p, loss_func, optimizer, train_dataloader, val_dataloader)


    # Test performance metrics
    rouge_metric = ROUGEScore() 
    bleu_metric = BLEUScore(n_gram= p.bleu_ngram)
    bleu, R1, R2, RL = performance_scores(model, test_dataloader, rouge_metric, bleu_metric)
    print(f"{'bleu':10}: {bleu:.3f} \n{'R1':10}: {R1:.3f}, \n{'R2':10}: {R2:.3f} \n{'RL':10}: {RL:.3f}")

    # Predict on test sentence
    n_predictions = 5
    for _ in range(n_predictions):
        prediction, target = predict_one(model, test_dataset)
        print(f"{'Prediction:':15} {prediction}") 
        print(f"{'Target:':15} {target}") 

if train_model:
    x = np.arange(p.n_epochs)
    plt.plot(x, train_history["train_loss"])
    plt.plot(x, train_history["bleu"])
    plt.plot(x, train_history["rouge1"])
    plt.plot(x, train_history["rouge2"])
    plt.plot(x, train_history["rougeL"])
    plt.legend(["train loss", "BLEU", "R1", "R2", "RL"])
    plt.ylim([0, 0.5])
    plt.title("Training and validation metrics")
    plt.xlabel("epoch")
    plt.ylabel("scores")
    plt.show()

