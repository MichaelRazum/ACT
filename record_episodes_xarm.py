import subprocess
import colorsys
from config.config import TASK_CONFIG
import os
import cv2
import h5py
import argparse
from time import  time, sleep

from teleopt import BusServoRemoteTelopt

import numpy as np

def get_color(diff):
    # Normalize the difference to a value between 0 and 1
    # Adjust the scaling factor (20 in this case) to make the color changes more noticeable
    normalized_diff = min(1, abs(diff) / 20)

    # Create a gradient from green (120) to yellow (60) to red (0)
    if normalized_diff < 0.5:
        # Green to Yellow
        hue = (1 - 2 * normalized_diff) * 60 / 360 + 60 / 360
    else:
        # Yellow to Red
        hue = (1 - normalized_diff) * 60 / 360

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)

    # Convert RGB to terminal color codes
    return f'\033[38;2;{int(r * 255)};{int(g * 255)};{int(b * 255)}m'


def print_with_diff(label1, list1, label2, list2):
    reset_color = '\033[0m'

    print(f"{label1}:", end=" ")
    for i, val in enumerate(list1):
        if i < len(list2):
            diff = val - list2[i]
            color = get_color(diff)
            print(f"{color}{val}{reset_color}", end=" ")
        else:
            print(val, end=" ")
    print()

    print(f"{label2}:", end=" ")
    for i, val in enumerate(list2):
        if i < len(list1):
            diff = val - list1[i]
            color = get_color(diff)
            print(f"{color}{val}{reset_color}", end=" ")
        else:
            print(val, end=" ")
    print()

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

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task2')
parser.add_argument('--num_episodes', type=int, default=30)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG

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
    cam = cv2.VideoCapture(find_webcam_index("C922 Pro Stream Webcam"))
    # Check if the camera opened successfully
    if not cam.isOpened():
        raise IOError("Cannot open camera")
    follower = BusServoRemoteTelopt('/dev/ttyUSB0')
    leader = BusServoRemoteTelopt('/dev/ttyUSB1')



    
    for i in range(num_episodes):
        # bring the follower to the leader and start camera
        os.system('espeak "Relase"')
        os.system(f'espeak "Epsides {i}')
        leader.disable_torque()
        for i in range(20):
            follower.set_goal_pos(leader.read_position())
            _ = capture_image(cam)

        os.system('espeak "adjust"')
        follower.enable_torque()
        leader.enable_torque()
        sleep(5)

        for _ in range(10):
            set_default_position(follower)
            set_default_position(leader)

        leader.disable_torque()

        os.system('espeak "position cube"')
        sleep(5)
        os.system('espeak "go"')
        # init buffers
        obs_replay = []
        action_replay = []
        for i in range(cfg['episode_len']):
            print(f'episode {i}/{cfg["episode_len"]}')
            # os.system(f'espeak "episode {i}"')
            # observation
            qpos = follower.read_position()
            qvel = follower.read_velocity()


            image = capture_image(cam)
            cv2.imshow('img', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            obs = {
                'qpos': pwm2pos(qpos),
                'qvel': pwm2vel(qvel),
                'images': {cn : image for cn in cfg['camera_names']}
            }
            # action (leader's position)
            action = leader.read_position()
            print_with_diff(label1='old', list1=qpos, label2='new', list2=action)
            # apply action
            max_diff = np.max(np.abs(np.array(qpos) - np.array(action)))
            # servo_runtime = 500
            servo_runtime = int(100 + 500 * min(max_diff,250) /250)
            print(f'episode {i}/{cfg["episode_len"]} {servo_runtime=}')
            follower.set_goal_pos(action, servo_runtime=servo_runtime)
            action = pwm2pos(action)
            # store data
            obs_replay.append(obs)
            action_replay.append(action)

        os.system('espeak "stop"')

        # disable torque
        #leader._disable_torque()
        #follower._disable_torque()

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
        data_dir = os.path.join(cfg['dataset_dir'], task)
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
    
    # leader.disable_torque()
    # follower.disable_torque()
