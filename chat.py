import random
import json

import torch

from model import NeuralNet
from commands import command_time, command_weather
from nltk_utils import bag_of_words, tokenize

device = torch.device("cpu")


def get_answer(sentence: str) -> str:
    with open("train_data.json", "r") as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    commands = {"time": command_time(), "weather": command_weather()}

    bot_name = "Bot"
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if (tag == intent["tag"]) and (tag in commands.keys()):
                return "{}: {}".format(bot_name, commands[tag])
            elif (tag == intent["tag"]) and (tag not in commands):
                return "{}: {}".format(bot_name, random.choice(intent["responses"]))

    else:
        return "{}: Я не смог тебя понять ...".format(bot_name)


if __name__ == "__main__":
    print("Напишите 'Выход' для завершения диалога")
    while True:
        sentence = input("User: ")
        if sentence in ("Выход", "выход"):
            break
        else:
            print(get_answer(sentence), flush=True)
