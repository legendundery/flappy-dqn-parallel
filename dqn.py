import argparse
import csv
import importlib
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.target_update_interval = 5000
        self.checkpoint_interval = 25000

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def configure_runtime(headless, disable_fps_limit):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ["FLAPPY_HEADLESS"] = "1"
    else:
        os.environ.pop("FLAPPY_HEADLESS", None)
    if disable_fps_limit:
        os.environ["FLAPPY_DISABLE_FPS_LIMIT"] = "1"
    else:
        os.environ.pop("FLAPPY_DISABLE_FPS_LIMIT", None)


def get_game_state_class(headless=False, disable_fps_limit=False):
    configure_runtime(headless=headless, disable_fps_limit=disable_fps_limit)
    module = importlib.import_module("game.flappy_bird")
    return module.GameState


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def image_to_tensor(image, device):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor).to(device)
    return image_tensor


def prepare_output_dirs():
    if not os.path.exists("pretrained_model"):
        os.mkdir("pretrained_model")
    if not os.path.exists("training_logs"):
        os.mkdir("training_logs")


def sync_target_network(model, target_model):
    target_model.load_state_dict(model.state_dict())


def append_log_row(log_path, row):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "episode", "episode_reward", "epsilon", "variant"])
        writer.writerow(row)


def save_checkpoint(model, variant, iteration):
    model_path = os.path.join("pretrained_model", variant + "_model_" + str(iteration) + ".pth")
    torch.save(model.state_dict(), model_path)


def train(model, start, use_target_network=False, log_interval=1000, headless=True):
    del headless
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    GameState = get_game_state_class(headless=True, disable_fps_limit=True)
    game_state = GameState()
    replay_memory = []

    variant = "target" if use_target_network else "baseline"
    log_path = os.path.join("training_logs", variant + "_reward_log.csv")

    target_model = None
    if use_target_network:
        target_model = NeuralNetwork().to(device)
        sync_target_network(model, target_model)
        target_model.eval()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
    action[0] = 1
    image_data, _, _ = game_state.frame_step(action.detach().cpu().numpy())
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data, device)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0
    episode = 0
    episode_reward = 0.0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:
        with torch.no_grad():
            output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
        random_action = random.random() <= epsilon
        action_index = random.randrange(model.number_of_actions) if random_action else int(torch.argmax(output).item())
        action[action_index] = 1

        image_data_1, reward_value, terminal = game_state.frame_step(action.detach().cpu().numpy())
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1, device)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action_batch_item = action.unsqueeze(0)
        reward = torch.tensor([[reward_value]], dtype=torch.float32, device=device)
        episode_reward += reward_value

        replay_memory.append((state.clone(), action_batch_item.clone(), reward.clone(), state_1.clone(), terminal))
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
        state_batch = torch.cat([d[0] for d in minibatch])
        action_batch = torch.cat([d[1] for d in minibatch])
        reward_batch = torch.cat([d[2] for d in minibatch]).squeeze(1)
        state_1_batch = torch.cat([d[3] for d in minibatch])
        terminal_batch = torch.tensor([d[4] for d in minibatch], dtype=torch.bool, device=device)

        with torch.no_grad():
            if use_target_network:
                output_1_batch = target_model(state_1_batch)
            else:
                output_1_batch = model(state_1_batch)
            next_q_value = torch.max(output_1_batch, dim=1).values
            y_batch = reward_batch + model.gamma * next_q_value * (~terminal_batch)

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        if use_target_network and (iteration + 1) % model.target_update_interval == 0:
            sync_target_network(model, target_model)

        state = state_1
        iteration += 1

        if terminal:
            episode += 1
            append_log_row(log_path, [iteration, episode, round(episode_reward, 4), round(float(epsilon), 6), variant])
            episode_reward = 0.0

        if iteration % model.checkpoint_interval == 0 or iteration == model.number_of_iterations:
            save_checkpoint(model, variant, iteration)

        if iteration % log_interval == 0:
            print(
                "variant:", variant,
                "iteration:", iteration,
                "elapsed time:", time.time() - start,
                "epsilon:", round(float(epsilon), 6),
                "reward:", reward_value,
                "Q max:", round(float(torch.max(output).item()), 6),
            )


def test(model):
    device = next(model.parameters()).device
    GameState = get_game_state_class(headless=False, disable_fps_limit=False)
    game_state = GameState()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
    action[0] = 1
    image_data, _, _ = game_state.frame_step(action.detach().cpu().numpy())
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data, device)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    with torch.no_grad():
        while True:
            output = model(state)[0]
            action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
            action_index = int(torch.argmax(output).item())
            action[action_index] = 1

            image_data_1, _, _ = game_state.frame_step(action.detach().cpu().numpy())
            image_data_1 = resize_and_bgr2gray(image_data_1)
            image_data_1 = image_to_tensor(image_data_1, device)
            state = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)


def load_model_for_test(model, variant, iteration):
    model_path = os.path.join("pretrained_model", variant + "_model_" + str(iteration) + ".pth")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("variant", nargs="?", default="baseline", choices=["baseline", "target"])
    parser.add_argument("iteration", nargs="?", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    cuda_is_available = torch.cuda.is_available()
    use_target_network = (args.variant == "target")

    model = NeuralNetwork()
    if args.iterations is not None:
        model.number_of_iterations = args.iterations

    if args.mode == "test":
        iteration = args.iteration if args.iteration is not None else model.number_of_iterations
        model = load_model_for_test(model, args.variant, iteration)
        if cuda_is_available:
            model = model.cuda()
        model.eval()
        test(model)
        return

    prepare_output_dirs()
    if cuda_is_available:
        model = model.cuda()
    model.apply(init_weights)
    start = time.time()
    train(model, start, use_target_network=use_target_network, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
