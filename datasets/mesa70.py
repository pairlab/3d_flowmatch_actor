from .rlbench import RLBenchDataset

MESA70_TASKS = [
    "apple_tray_on",
    "avocado_basket_contain_region",
    "bagel_cutting_board_on",
    "bar_cabinet_bottom_region_and_close",
    "bar_soap_cabinet_top_region_and_close",
    "beer_slide_cabinet_contain_region_and_close",
    "bell_pepper_sliding_top_box_contain_region_and_close",
    "bottled_water_tray_on",
    "bowl_microwave_heating_region_and_close",
    "broccoli_pan_on",
    "candle_tray_on",
    "carrot_bowl_on",
    "cheese_plate_on",
    "corn_cutting_board_on",
    "croissant_slide_cabinet_contain_region_and_close",
    "cucumber_cabinet_bottom_region_and_close",
    "cup_bowl_drainer_right_region",
    "egg_bowl_on",
    "eggplant_cutting_board_on",
    "fish_pan_on",
    "fish_plate_on",
    "garlic_cabinet_middle_region_and_close",
    "garlic_pan_on",
    "jam_basket_contain_region",
    "jug_basket_contain_region",
    "kiwi_cabinet_top_region_and_close",
    "lemon_plate_on",
    "lime_bowl_on",
    "mango_sliding_top_box_contain_region_and_close",
    "mug_bowl_drainer_left_region",
    "mushroom_bowl_on",
    "onion_tray_on",
    "open_and_avocado_cabinet_top_region",
    "open_and_book_slide_cabinet_contain_region",
    "open_and_broccoli_cabinet_bottom_region",
    "open_and_candle_slide_cabinet_contain_region",
    "open_and_canned_food_microwave_heating_region",
    "open_and_can_sliding_top_box_contain_region",
    "open_and_carrot_cabinet_top_region",
    "open_and_cheese_sliding_top_box_contain_region",
    "open_and_eggplant_microwave_heating_region",
    "open_and_lemon_cabinet_middle_region",
    "open_and_mushroom_cabinet_bottom_region",
    "open_and_onion_cabinet_middle_region",
    "open_and_peach_cabinet_top_region",
    "open_and_squash_microwave_heating_region",
    "open_and_tomato_left_slide_cabinet_contain_region",
    "open_and_water_bottle_left_slide_cabinet_contain_region",
    "open_and_wine_slide_cabinet_contain_region",
    "orange_cutting_board_on",
    "orange_plate_on",
    "peach_basket_contain_region",
    "peach_cutting_board_on",
    "peach_tray_on",
    "plate_bowl_drainer_left_region",
    "plate_bowl_drainer_right_region",
    "potato_microwave_heating_region_and_close",
    "rolling_pin_left_bowl_drainer_right_region",
    "rolling_pin_left_pan_on",
    "rolling_pin_left_tray_on",
    "sponge_cabinet_middle_region_and_close",
    "squash_pan_on",
    "squash_tray_on",
    "tomato_left_cutting_board_on",
    "tomato_left_pan_on",
    "tomato_left_tray_on",
    "water_bottle_left_basket_contain_region",
    "water_bottle_left_tray_on",
    "wine_sliding_top_box_contain_region",
    "wine_tray_on",
    "yogurt_basket_contain_region",
]


class Mesa70Dataset(RLBenchDataset):
    """Mesa-70 single-arm dataset (single-task quick test, apple_tray_on)."""
    tasks = ["apple_tray_on"]
    cameras = ("leftshoulder", "rightshoulder", "robot0_eye_in_hand")
    camera_inds = None
    camera_inds2d = None
    train_copies = 50


class Mesa70MultiTaskDataset(RLBenchDataset):
    """Mesa-70 single-arm dataset (all 71 tasks)."""
    tasks = MESA70_TASKS
    cameras = ("leftshoulder", "rightshoulder", "robot0_eye_in_hand")
    camera_inds = None
    camera_inds2d = None
    train_copies = 10
