from .rlbench import RLBenchDataset


MESA_TASKS = [
    "apple_tray_on",
    "bottled_water_tray_on",
    "broccoli_pan_on",
    "candle_tray_on",
    "carrot_bowl_on",
    "cheese_plate_on",
    "corn_cutting_board_on",
    "egg_bowl_on",
    "eggplant_cutting_board_on",
    "garlic_pan_on",
    "jam_basket_contain_region",
    "jug_basket_contain_region",
    "lemon_plate_on",
    "lime_bowl_on",
    "mushroom_bowl_on",
    "onion_tray_on",
    "orange_plate_on",
]


class MesaBimanualDataset(RLBenchDataset):
    """Mesa bimanual dataset (single-task debug subset)."""
    tasks = ["apple_tray_on"]
    cameras = ("front", "wrist_right", "wrist_left")
    camera_inds = None
    camera_inds2d = None
    train_copies = 100


class MesaBimanualMultiTaskDataset(RLBenchDataset):
    """Mesa bimanual dataset (17-task overfit)."""
    tasks = MESA_TASKS
    cameras = ("front", "wrist_right", "wrist_left")
    camera_inds = None
    camera_inds2d = None
    train_copies = 10
