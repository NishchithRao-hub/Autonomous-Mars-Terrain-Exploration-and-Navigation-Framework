DEFAULT_CONFIG = {
    # ======== TOPOLOGY =======
    #  general configuration for the topology of operational area
    "initial": [0, 0],
    "size": [21, 21],
    #  configuration regarding the movements of uav
    "movementCost": 0.2,

    # ======== ENVIRONMENT =======
    # configuration regarding the random map generation
    # absolute number of obstacles, randomly placed in env
    "obstacles": 14,
    # if rows/columns activated the obstacles will be placed in a semi random spacing
    "number_rows": None,
    "number_columns": None,
    # noise activated only when row/columns activated
    # maximum noise on each axes
    "noise": [0, 0],
    # margins expressed in cell if rows/columns not activated
    "margins": [1, 1],
    # obstacle size expressed in cell if rows/columns not activated
    "obstacle_size": [2, 2],
    # max number of steps for the environment
    "max_steps": 800,
    "collision_reward": -500,
    "out_of_bounds_reward": -500,
    "99%_bonus_reward": 400,
    "25%_bonus_reward": 50,
    "50%_bonus_reward": 100,
    "75%_bonus_reward": 150,

    # ======== SENSORS | LIDAR =======
    "lidar_range": 6,
    "lidar_channels": 32,

    # ======== VIEWER =========
    "viewer": {"width": 21 * 30,
               "height": 21 * 30,
               "title": "Mars-Explorer-V1",
               "drone_img": 'mars-explorer/tests/img/drone.png',
               "obstacle_img": 'mars-explorer/tests/img/block.png',
               "background_img": 'mars-explorer/tests/img/mars.jpg',
               "light_mask": "mars-explorer/tests/img/light_350_hard.png",
               "night_color": (20, 20, 20),
               "draw_lidar": True,
               "draw_grid": False,
               "draw_traceline": False
               }
}
