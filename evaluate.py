import subprocess

import h5py

from ACT.training.utils import make_policy, vel2pwm, get_image
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import cv2
import torch
import pickle
import argparse
from time import time, sleep

from teleopt import BusServoRemoteTelopt
import numpy as np

def set_default_position(servo):
    l = [122, 0, 0, 0, 0, 0]
    for n, v in enumerate(l):
        servo.run(n + 1, v, 500)




def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi*1.33, pi*1.33]
    """
    return (pwm / 500 - 1) * 4.19

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities
    """
    return pwm * 4.19 / 500

def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return (pos /  4.19 + 1.) * 500

def set_default_position(servo):
    l = [80, 500, 500, 500, 500, 500]
    for n, pwm in enumerate(l):
        pos = pwm2pos(pwm)
        v  = pos2pwm(pos)

        servo.run(n + 1, int(v), 800)

def find_webcam_index(device_name):
    command = "v4l2-ctl --list-devices"
    output = subprocess.check_output(command, shell=True, text=True)
    devices = output.split('\n\n')

    for device in devices:
        #print(device)
        if device_name in device:
            lines = device.split('\n')
            for line in lines:
                if "video" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith('/dev/video'):
                            return (part)


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task2')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']


def capture_image(cam):
    # Capture a single frame
    _, frame = cam.read()
    # Generate a unique filename with the current date and time
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('asdf', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)
    # cv2.imshow('asdf', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

if __name__ == "__main__":
    # init camera
    # cam = cv2.VideoCapture(cfg['camera_port'])
    cam = cv2.VideoCapture(find_webcam_index("C922 Pro Stream Webcam"))
    # Check if the camera opened successfully
    if not cam.isOpened():
        raise IOError("Cannot open camera")
    # init follower
    # follower = Robot(device_name=ROBOT_PORTS['follower'])
    follower = BusServoRemoteTelopt('/dev/ttyUSB1')

    os.system('espeak "setting follower"')
    follower.enable_torque()
    sleep(5)

    for _ in range(10):
        set_default_position(follower)

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # bring the follower to the leader
    for i in range(90):
        follower.read_position()
        _ = capture_image(cam)
    
    obs = {
        'qpos': pwm2pos(follower.read_position()),
        'qvel': vel2pwm(follower.read_velocity()),
        'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
    }
    os.system('say "start"')

    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            for t in range(cfg['episode_len']):
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = pos2pwm(action).astype(int)
                ### take action
                pos_old = follower.read_position()
                max_diff = np.max(np.abs(np.array(pos_old) - np.array(action)))
                servo_runtime = int(100 + 500 * min(max_diff, 250) / 250)
                follower.set_goal_pos(action, servo_runtime=servo_runtime)

                ### update obs
                obs = {
                    'qpos': pwm2pos(follower.read_position()),
                    'qvel': vel2pwm(follower.read_velocity()),
                    'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)

        os.system('say "stop"')

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        data_dir = cfg['dataset_dir']  
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        with h5py.File("data/demo/trained.hdf5", 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
    
    # disable torque
    follower._disable_torque()