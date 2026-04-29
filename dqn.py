import argparse
import csv
import importlib
import os
import queue
import random
import time
from collections import deque
from multiprocessing import get_context

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
        self.worker_sync_interval = 1000
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


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    return image_data.astype(np.uint8)


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


def state_to_tensor(state_array, device):
    state_tensor = torch.from_numpy(state_array.astype(np.float32))
    if state_tensor.ndim == 3:
        state_tensor = state_tensor.unsqueeze(0)
    return state_tensor.to(device)


def build_initial_state(game_state):
    action = np.zeros(2, dtype=np.float32)
    action[0] = 1
    image_data, _, _ = game_state.frame_step(action)
    frame = resize_and_bgr2gray(image_data)
    stacked = np.stack([frame, frame, frame, frame], axis=0)
    return stacked


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


def build_state_packet(model, epsilon):
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {"state_dict": cpu_state, "epsilon": float(epsilon)}


def broadcast_worker_state(weight_queues, packet):
    for worker_queue in weight_queues:
        try:
            while True:
                worker_queue.get_nowait()
        except queue.Empty:
            pass
        worker_queue.put(packet)


def actor_worker(worker_id, variant, total_iterations, initial_packet, experience_queue, weight_queue, stop_event):
    del worker_id, variant, total_iterations
    torch.set_num_threads(1)
    GameState = get_game_state_class(headless=True, disable_fps_limit=True)

    model = NeuralNetwork()
    model.load_state_dict(initial_packet["state_dict"])
    model.eval()
    epsilon = initial_packet["epsilon"]

    game_state = GameState()
    state = build_initial_state(game_state)
    episode_reward = 0.0

    while not stop_event.is_set():
        try:
            while True:
                packet = weight_queue.get_nowait()
                model.load_state_dict(packet["state_dict"])
                model.eval()
                epsilon = packet["epsilon"]
        except queue.Empty:
            pass

        state_tensor = state_to_tensor(state, device="cpu")
        with torch.no_grad():
            output = model(state_tensor)[0]

        random_action = random.random() <= epsilon
        action_index = random.randint(0, model.number_of_actions - 1) if random_action else int(torch.argmax(output).item())
        action = np.zeros(model.number_of_actions, dtype=np.float32)
        action[action_index] = 1

        image_data_1, reward_value, terminal = game_state.frame_step(action)
        next_frame = resize_and_bgr2gray(image_data_1)
        next_state = np.concatenate((state[1:, :, :], next_frame[np.newaxis, :, :]), axis=0)

        experience_queue.put({
            "type": "transition",
            "state": state,
            "action": action_index,
            "reward": float(reward_value),
            "next_state": next_state,
            "terminal": bool(terminal),
        })

        episode_reward += reward_value
        if terminal:
            experience_queue.put({
                "type": "episode",
                "episode_reward": float(episode_reward),
            })
            episode_reward = 0.0

        state = next_state


def sample_minibatch(replay_memory, batch_size):
    minibatch = random.sample(replay_memory, min(len(replay_memory), batch_size))
    states = np.stack([item["state"] for item in minibatch], axis=0)
    actions = np.array([item["action"] for item in minibatch], dtype=np.int64)
    rewards = np.array([item["reward"] for item in minibatch], dtype=np.float32)
    next_states = np.stack([item["next_state"] for item in minibatch], axis=0)
    terminals = np.array([item["terminal"] for item in minibatch], dtype=np.bool_)
    return states, actions, rewards, next_states, terminals


def save_checkpoint(model, variant, iteration):
    model_path = os.path.join("pretrained_model", variant + "_model_" + str(iteration) + ".pth")
    torch.save(model.state_dict(), model_path)


def train(model, start, use_target_network=False, num_workers=1, log_interval=100):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    replay_memory = deque(maxlen=model.replay_memory_size)

    variant = "target" if use_target_network else "baseline"
    log_path = os.path.join("training_logs", variant + "_reward_log.csv")

    target_model = None
    if use_target_network:
        target_model = NeuralNetwork().to(device)
        sync_target_network(model, target_model)
        target_model.eval()

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    current_epsilon = model.initial_epsilon

    ctx = get_context("spawn")
    experience_queue = ctx.Queue(maxsize=max(512, num_workers * 128))
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(num_workers)]
    stop_event = ctx.Event()

    initial_packet = build_state_packet(model, current_epsilon)
    workers = []
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=actor_worker,
            args=(worker_id, variant, model.number_of_iterations, initial_packet, experience_queue, weight_queues[worker_id], stop_event),
        )
        process.start()
        workers.append(process)

    global_episode = 0
    iteration = 0
    latest_reward = 0.0

    try:
        while iteration < model.number_of_iterations:
            item = experience_queue.get()

            if item["type"] == "episode":
                global_episode += 1
                append_log_row(log_path, [iteration, global_episode, item["episode_reward"], float(current_epsilon), variant])
                continue

            replay_memory.append(item)
            iteration += 1
            current_epsilon = epsilon_decrements[iteration - 1]
            latest_reward = item["reward"]

            states, actions, rewards, next_states, terminals = sample_minibatch(replay_memory, model.minibatch_size)
            state_batch = state_to_tensor(states, device)
            next_state_batch = state_to_tensor(next_states, device)
            action_batch = torch.from_numpy(actions).to(device)
            reward_batch = torch.from_numpy(rewards).to(device)
            terminal_batch = torch.from_numpy(terminals).to(device)

            with torch.no_grad():
                if use_target_network:
                    output_1_batch = target_model(next_state_batch)
                else:
                    output_1_batch = model(next_state_batch)
                next_q_value = torch.max(output_1_batch, dim=1).values
                y_batch = reward_batch + model.gamma * next_q_value * (~terminal_batch)

            q_value = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            optimizer.zero_grad()
            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()

            if use_target_network and iteration % model.target_update_interval == 0:
                sync_target_network(model, target_model)

            if iteration % model.worker_sync_interval == 0:
                packet = build_state_packet(model, current_epsilon)
                broadcast_worker_state(weight_queues, packet)

            if iteration % model.checkpoint_interval == 0 or iteration == model.number_of_iterations:
                save_checkpoint(model, variant, iteration)

            if iteration % log_interval == 0:
                with torch.no_grad():
                    output = model(state_batch[:1])[0]
                print(
                    "variant:", variant,
                    "iteration:", iteration,
                    "elapsed time:", time.time() - start,
                    "epsilon:", current_epsilon,
                    "reward:", latest_reward,
                    "Q max:", float(torch.max(output).item()),
                    "workers:", num_workers,
                )
    finally:
        stop_event.set()
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()


def test(model):
    device = next(model.parameters()).device
    GameState = get_game_state_class(headless=False, disable_fps_limit=False)
    game_state = GameState()
    state = build_initial_state(game_state)

    with torch.no_grad():
        while True:
            state_tensor = state_to_tensor(state, device)
            output = model(state_tensor)[0]
            action_index = int(torch.argmax(output).item())
            action = np.zeros(model.number_of_actions, dtype=np.float32)
            action[action_index] = 1

            image_data_1, _, _ = game_state.frame_step(action)
            next_frame = resize_and_bgr2gray(image_data_1)
            state = np.concatenate((state[1:, :, :], next_frame[np.newaxis, :, :]), axis=0)


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
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--log-interval", type=int, default=100)
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
    train(model, start, use_target_network=use_target_network, num_workers=args.num_workers, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
