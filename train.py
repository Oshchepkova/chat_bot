import numpy as np
import json
import nltk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words
from model import NeuralNet


from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # индексация - dataset[i] для получения i-го набора
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # для получения размера набора данных
    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    FINAL_FILE = "data.pth"

    with open("train_data.json", "r") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    # цикл по всем фразам из набора
    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            # предложение -> массив слов/пунктационных символов
            words = nltk.word_tokenize(pattern)
            all_words.extend(words)
            xy.append((words, tag))

    # стеммируем слова - оставляя только основу слова, удаляем знаки пунктуации
    ignore_words = ["?", ".", "!", ","]
    all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
    # удаляем дублирующие слова, сортируем
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print(len(tags), "теги:", tags)
    print(len(all_words), "уникальные слова:", all_words)

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X: список слов из каждого предложения в наборе данных
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: функции CrossEntropyLoss в PyTorch необходима только метка класса
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # параметры для обучения
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)

    dataset = ChatDataset()
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    device = torch.device("cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # обучение модели
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )

    print("final loss: {:.4f}".format(loss.item()))

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    torch.save(data, FINAL_FILE)

    print("Обучение завершено. Файл сохранен: {}".format(FINAL_FILE))
