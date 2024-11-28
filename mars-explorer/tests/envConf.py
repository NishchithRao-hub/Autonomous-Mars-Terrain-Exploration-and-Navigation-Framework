# import gym
# import numpy as np
# import math
# import time
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# plotstyle="ggplot"
# plt.style.use(f"{plotstyle}")
#
# from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
#
# SAVE = True
#
# size = [81,81]
#
# conf["initial"] = [int(size[0]/2), int(size[1]/2 )]
# conf["size"] = size
# conf["obstacles"] = 4
# conf["obstacle_size"] = [1,1]
# conf["lidar_range"] = 6
#
# conf["viewer"]["width"] = size[0]*30
# conf["viewer"]["height"] = size[1]*30
# conf["viewer"]["night_color"] = (0,0,0)
# conf["viewer"]["draw_lidar"] = True
#
# def saveRend(rend):
#     plt.rcParams["axes.grid"] = False
#     plt.axis('off')
#     plt.imshow(rend)
#     plt.savefig(f"{step}_env.png", bbox_inches='tight')
#     plt.close()
#
# def plotHeatMap(img):
#     fig, ax = plt.subplots(1,1)
#     h = sns.heatmap(img[:,:,0],
#                     annot=True,
#                     linewidths=.5)
#
#     h.set(yticks=[])
#     h.set(xticks=[])
#
#     ax.set_ylabel('')
#     ax.set_xlabel('')
#
#     if SAVE:
#         plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
#     else:
#         plt.show()
#     plt.close()
#
# env = gym.make('mars_explorer:exploConf-v1', conf=conf)
# observation = env.reset()
#
# for step in range(3):
#
#     img = env.reset()
#     rend = env.render()
#
#     if SAVE:
#         saveRend(rend)
#
#     plotHeatMap(np.swapaxes(img, 0, 1))
#
# env.close()

import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

# General plotting settings
plotstyle = "ggplot"
plt.style.use(plotstyle)

# Save option
SAVE = True

# Configure environment
size = [81, 81]

conf["initial"] = [int(size[0] / 2), int(size[1] / 2)]
conf["size"] = size
conf["obstacles"] = 4
conf["obstacle_size"] = [1, 1]
conf["lidar_range"] = 6

conf["viewer"]["width"] = size[0] * 30
conf["viewer"]["height"] = size[1] * 30
conf["viewer"]["night_color"] = (0, 0, 0)
conf["viewer"]["draw_lidar"] = True


# Helper functions
def saveRend(rend):
    """Save the rendered environment to a file."""
    plt.rcParams["axes.grid"] = False
    plt.axis('off')
    plt.imshow(rend)
    plt.savefig(f"{step}_env.png", bbox_inches='tight')
    plt.close()


def plotHeatMap(img):
    """Plot a heatmap of the environment observation."""
    # Ensure img is a 2D array
    if img.ndim == 3:  # If it's 3D, use the first channel
        img = img[:, :, 0]

    fig, ax = plt.subplots(1, 1)
    h = sns.heatmap(img,
                    annot=False,  # Disable annotations for large arrays
                    linewidths=.5,
                    cbar=True)  # Enable colorbar for context

    h.set(yticks=[])
    h.set(xticks=[])

    ax.set_ylabel('')
    ax.set_xlabel('')

    if SAVE:
        plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# Initialize the environment
env = gym.make('mars_explorer:exploConf-v1', conf=conf)
observation = env.reset()

for step in range(3):
    # Reset environment and get observation tuple
    obs = env.reset()  # `obs` is a tuple: (array, metadata_dict)

    img, metadata = obs  # Unpack the tuple
    print(f"Step {step}: Observation type={type(img)}, Metadata={metadata}")

    # Ensure img is a NumPy array
    if not isinstance(img, np.ndarray):
        print(f"Error: Observation is not a valid NumPy array.")
        continue

    rend = env.render()  # Render the environment

    if SAVE:
        saveRend(rend)  # Save the rendered image

    plotHeatMap(img)  # Generate and save the heatmap

env.close()


