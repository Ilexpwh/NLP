import time
import torch
import torch.nn as nn
import torchvision
from random import randint
from torch.utils.data import DataLoader
from tqdm import tqdm
from Params import Params
from collections import defaultdict
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
from flickerdata import FlickerData


class Encoder(torch.nn.Module):
    """ Encodes image into an embeding vector used for decoder """

    def __init__(self, p: Params):
        super().__init__()

        self.vgg = torchvision.models.vgg19()  # load in encoder
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p= p.dropout_rate)
        for _, param in self.vgg.named_parameters():  # freeze vgg weights
            param.requires_grad = False

        self.latent_embedding = nn.Linear(4096, p.emb_dim)  # maps latent_vector to embed_vector

    def forward(self, X):
        """ Takes in image X and outputs embed_vector """

        batch_size = X.shape[0]

        X = self.vgg.features(X)  # convolutional part
        X = self.vgg.avgpool(X)  # sets to fixed size (512,7,7)
        X = X.view(batch_size, 512 * 7 * 7)  # (batch,25088) flatten
        X = self.vgg.classifier[0:3](X)  # (batch,4096)
        X = self.latent_embedding(X)  # (batch,emb_dim)
        X = self.dropout(self.relu(X)) 
        return X


class Decoder(torch.nn.Module):
    """ Decodes a embed_vector into lemmas/captions """

    def __init__(self, p: Params):
        super().__init__()

        self.p = p
        self.lstm = nn.LSTM(
            input_size=p.emb_dim,
            hidden_size=p.hidden_size,
            num_layers=p.n_layers,
            batch_first=True)
        self.dropout = nn.Dropout(p.dropout_rate)
        self.word_emb = nn.Embedding(p.vocab_size, p.emb_dim)
        self.output_emb = nn.Linear(p.hidden_size, p.vocab_size)

    def forward(self, embed_vector, caption):
        """ Takes in embed_vector and captions and outputs prob distribution 
            Input:
                embed_vector (batch, emb_dim)
                caption (batch, max_sen_len)
            Output:
                probability (batch, max_sen_len, vocab_size)
        """
        batch_size = embed_vector.shape[0]
        embed_captions = self.dropout(self.word_emb(caption))  # (batch_size, max_sen_len, emb_dim)
        inputs = torch.cat(
            (embed_vector.view(batch_size, 1, self.p.emb_dim), embed_captions),
            dim=1)  # (batch_size, max_sen_len+1, emb_dim)
        output, _ = self.lstm(inputs)  # (batch_size, max_sen_len+1, hidden_size)
        logits = self.output_emb(output)  # (batch_size, max_sen_len+1, vocab_size)
        return logits

    def predict(self, latent_embedding, vocab):
        """ Used to predict caption for images during testing 
            
            Input:
              latent_embedding (batch_size, emb_dim) -> batch_size number of sentences
            Output:
              List of integer encoded sentences
        """
        n_sentences = latent_embedding.shape[0]
        sentences = [] # list of torch tensors

        for sen_idx in range(n_sentences):

            sentence = []
            
            # first token
            first_token = torch.unsqueeze(latent_embedding[sen_idx,:], dim= 0).to(self.p.device) # (1, emb_dim)  
            init_tuple = (torch.zeros((self.p.n_layers, self.p.hidden_size)).to(self.p.device),
                          torch.zeros((self.p.n_layers, self.p.hidden_size)).to(self.p.device)) # (n_layers, emb_dim)
            lstm_out, (h_t, c_t) = self.lstm(first_token, init_tuple) # (1, hidden_size) 
            prob = self.output_emb(lstm_out) # (1, vocab_size)
            norm_prob = nn.functional.softmax(prob, dim= 1) # (1, vocab_size)

            # Sample next token
            next_token = self.p.sampling_strat(norm_prob)
            emb_next_token = torch.unsqueeze(self.word_emb(next_token), dim= 0)
            sentence.append(next_token.item())

            # while loop until EOS or max_sen_len
            while len(sentence) < self.p.max_pred_sen:

                #next_input = torch.unsqueeze(emb_next_token, dim= 0)
                lstm_out, (h_t, c_t) = self.lstm(emb_next_token, (h_t, c_t)) 
                prob = self.output_emb(lstm_out)
                norm_prob = nn.functional.softmax(prob, dim= 1)
                next_token = self.p.sampling_strat(norm_prob)
                emb_next_token = torch.unsqueeze(self.word_emb(next_token), dim= 0)
                sentence.append(next_token.item())

                if next_token.item() == vocab.get("<EOS>"):
                    break

            sentences.append(torch.tensor(sentence, dtype= torch.int32))

        return sentences


class ImageCaptionGenerator(torch.nn.Module):

    def __init__(self, vocab_lc, p: Params):
        super().__init__()

        # parameters and vocab
        self.vocab_lc = vocab_lc
        self.p = p

        # set up network layers
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)

    def forward(self, image, lemma):
        """ Takes in images and returns probability distribution for each token """
        latent_embedding = self.encoder(image)
        return self.decoder(latent_embedding, lemma)

    def predict(self, image):
        """ Takes in image and returns caption """
        latent_image = self.encoder(image)
        return self.decoder.predict(latent_image, self.vocab_lc)

def performance_scores(model: ImageCaptionGenerator, data_loader: DataLoader, rouge_metric, bleu_metric):
    """ Calculates performance metrics from dataloader 
        
        Input:
          - model
          - Dataloader
          - rouge metric
          - bleu metric
        Output:
         average scores: (bleu, rouge1, rouge2, rouge3) 
    """
    rouge1 = 0
    rouge2 = 0
    rougeL = 0
    bleu = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):


            # images: [batch,channel,width,height], lemmas: [batch,max_sen_len]
            images, lemmas, _ = batch
            batch_size = images.shape[0]
            images = images.to(model.p.device)

            # forward pass
            predicted_lemmas = model.predict(images)

            # calculate ROUGE and BLEU
            for sen_idx in range(batch_size):
                
                # prediction
                predicted_lemma = data_loader.dataset.decode_lc(predicted_lemmas[sen_idx])
                try:
                    EOS_idx = predicted_lemma.index("<EOS>")
                    predicted_string = " ".join(predicted_lemma[1:EOS_idx])
                except:
                    predicted_string = " ".join(predicted_lemma[1:])

                # target
                EOS_idx = (lemmas[sen_idx] == model.vocab_lc.get("<EOS>")).nonzero().item()
                decoded_target = data_loader.dataset.decode_lc(lemmas[sen_idx, 1:EOS_idx])
                target = " ".join(decoded_target)

                # BLEU
                bleu += bleu_metric([predicted_string], [[target]]).item()

                # Rouge
                rouge_scores = rouge_metric(predicted_string, target)
                rouge1 += rouge_scores["rouge1_recall"].item()
                rouge2 += rouge_scores["rouge2_recall"].item()
                rougeL += rouge_scores["rougeL_recall"].item()


    # Mean
    n_sentences = len(data_loader.dataset)
    bleu /= n_sentences
    rouge1 /= n_sentences
    rouge2 /= n_sentences
    rougeL /= n_sentences

    return (bleu, rouge1, rouge2, rougeL)


def train_ICG(
    model: ImageCaptionGenerator,
    p: Params,
    loss_func: torch.nn,
    optimizer: torch.optim ,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader):
    
    # Local help variables
    device = p.device
    PAD = model.vocab_lc["<PAD>"]
    rouge_metric = ROUGEScore() 
    bleu_metric = BLEUScore(n_gram= p.bleu_ngram)

    # Contains the statistics that will be returned.
    history = defaultdict(list)

    # Main training loop
    print(f" Started training for {p.n_epochs} epochs, n_batches = {len(train_dataloader)}, using device: {p.device}")
    progress = tqdm(range(p.n_epochs), 'Training')
    for epoch in progress:

        t0 = time.time()

        # run one epoch of training
        model.train()
        training_loss = 0
        for _, batch in enumerate(train_dataloader):

            # images: [batch,channel,width,height], lemmas: [batch,max_sen_len]
            images, lemmas, _ = batch
            images = images.to(device)
            lemmas = lemmas.to(device) # (batch_size, max_sen_len)

            # set target to padded lemmas
            target = torch.cat((lemmas, torch.tensor(PAD).repeat(p.batch_size, 1).to(p.device)), dim=1) # (batch_size, max_sen_len+1)
            target = torch.flatten(target)  # flatten batch dims (batch_size*(max_sen_len+1),)

            # forward pass
            output_logits = model(images, lemmas) # (batch_size, max_sen_len+1, vocab_size)
            output_logits = torch.flatten(output_logits, start_dim=0, end_dim=1)  # (batch_size*(max_sen_len+1), vocab_size)

            # calculate loss
            loss = loss_func(output_logits, target)
            training_loss += loss.item()

            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss /= len(train_dataloader.dataset) # avg w.r.t number of images

        # Validation loop
        bleu, R1, R2, RL = performance_scores(model, dev_dataloader, rouge_metric, bleu_metric)

        t1 = time.time()

        # Save epoch data
        history['epoch'].append(epoch)
        history['time'].append(t1 - t0)
        history['train_loss'].append(training_loss)
        history['bleu'].append(bleu)
        history['rouge1'].append(R1)
        history['rouge2'].append(R2)
        history['rougeL'].append(RL)

        progress.set_postfix({
            'time': f'{t1 - t0:.1f}',
            'train_loss': f'{training_loss:.3f}',
            'bleu': f'{bleu:.3f}',
            'rouge1': f'{R1:.3f}',
            'rouge2': f'{R2:.3f}',
            'rougeL': f'{RL:.3f}'})
        torch.save(model.state_dict(), "Models/icg.pt")  # better to save model that performed best on validation set
    
    return history


def predict_one(model: ImageCaptionGenerator, dataset: FlickerData, idx: int = None):
    """ Performs one prediction on random image in dataset"""

    model.eval()
    if idx is None:
        sample_idx = randint(0, len(dataset)-1)
    image, lemma, _ = dataset[sample_idx]
    image = torch.unsqueeze(image, dim=0).to(model.p.device) 
    model_out = model.predict(image)

    token_decoder = dataset.decode_lc

    # Prediction
    decoded_prediction = token_decoder(model_out[0])
    try:
        EOS_pred = decoded_prediction.index("<EOS>")
    except:
        EOS_pred = -1
    if EOS_pred == -1:
        prediction = " ".join(decoded_prediction[1:])
    else:
        prediction = " ".join(decoded_prediction[1:EOS_pred])

    # Target
    decoded_lemma = token_decoder(lemma)
    EOS_target = decoded_lemma.index("<EOS>")
    target = " ".join(decoded_lemma[1:EOS_target])

    return (prediction, target)