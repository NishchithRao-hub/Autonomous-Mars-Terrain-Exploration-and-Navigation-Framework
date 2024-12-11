import gym
import time
import argparse
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf


def get_conf():
    conf["size"] = [30, 30]
    conf["obstacles"] = 20
    conf["lidar_range"] = 4
    conf["obstacle_size"] = [1, 3]
    conf["viewer"]["night_color"] = (0, 0, 0)
    conf["viewer"]["draw_lidar"] = True
    conf["viewer"]["drone_img"] = "mars-explorer/tests/img/drone.png"
    conf["viewer"]["obstacle_img"] = "mars-explorer/tests/img/block.png"
    conf["viewer"]["background_img"] = "mars-explorer/tests/img/mars.jpg"
    conf["viewer"]["light_mask"] = "mars-explorer/tests/img/light_350_hard.png"
    return conf


def getArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-w', '--warm-up',
        default=0,
        type=int,
        help='Number of warm up games ')
    argparser.add_argument(
        '-g', '--games',
        default=10,
        type=int,
        help='Games to be played')
    argparser.add_argument(
        '-s', '--save',
        default=False,
        action="store_true",
        help='Save each rendered image')
    return argparser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    conf = get_conf()

    env = gym.make('mars_explorer:exploConf-v01', conf=conf)
    observation, info = env.reset()
    for _ in range(20):
        env.render()
        action = env.action_space.sample()  # agent here takes random actions
        observation, reward, done, truncated, info = env.step(action)
        if done:
            observation = env.reset()
        time.sleep(0.3)
    env.close()
