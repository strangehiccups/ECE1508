from importlib import reload
import torch
import matplotlib.pyplot as plt

from DataLoader import LJSpeechDataset, collate_fn
import utils
from DeepSpeech2 import DeepSpeech2


# load LJSpeech dataset
dataset = LJSpeechDataset("../data/LJSpeech-1.1")
num_samples = len(dataset)
print(f"Number of samples in the dataset: {num_samples}")
if num_samples > 0:
    sample = dataset[0]
    print(f"Sample text: {sample.raw_text}")
    print(f"Sample audio shape: {sample.raw_audio.shape}")
else:
    print("No samples found in the dataset. Check your data directory and metadata.csv.")
# visualise a sample
utils.plot_audio_mel_spectrogram(sample.mel_audio, sample_rate=sample.sample_rate)
print(f"Raw text: {sample.raw_text}")
print(f"Tokenized text: {sample.tokenized_text}")

# split into train-validation-test
train_size = int(num_samples*0.8)
val_size = int(num_samples*0.1)
test_size = num_samples - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],
                                                             generator=torch.Generator().manual_seed(1508)) # for reproducibility

# initialise loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


# Training loop
learningRate = 3e-4
max_epochs = 1

specs = next(iter(train_loader))['padded_spectrograms']
in_channels = specs.shape[1]

model = DeepSpeech2(conv_in_channels=in_channels)
optimiser = torch.optim.AdamW(params=model.parameters(),
                              lr=learningRate)
utils.train(model=model,
            optimiser=optimiser,
            train_loader=train_loader,
            loss_fn=model.loss_fn,
            max_epochs=max_epochs)