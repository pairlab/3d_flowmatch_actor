import numpy as np

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


class MesaBimanualMultiTaskSingleDataset(MesaBimanualMultiTaskDataset):
    """Single-task filtered view of the multitask v2 zarr.

    Inherits the 17-task `tasks` list so task_id / instruction indexing keeps
    working, but restricts the sample set to entries whose task_id is in
    `task_filter`. Used to isolate demo-scale vs task-diversity as the trigger
    for proprio collapse in the multitask plateau investigation.
    """
    task_filter = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.task_filter, (
            "MesaBimanualMultiTaskSingleDataset subclasses must set "
            "`task_filter` to a non-empty tuple of task names."
        )
        tid_keep = {self.tasks.index(t) for t in self.task_filter}
        tids_all = np.asarray(self.annos['task_id'][:])
        n_total_chunks = len(tids_all) // self.chunk_size
        self._filter_chunks = np.asarray([
            i for i in range(n_total_chunks)
            if int(tids_all[i * self.chunk_size]) in tid_keep
        ], dtype=np.int64)
        print(
            f"Filtered {len(self._filter_chunks)}/{n_total_chunks} "
            f"chunks to tasks {list(self.task_filter)}"
        )

    def _n_chunks(self):
        return len(self._filter_chunks)

    def _resolve_idx(self, idx):
        chunk_i = int(self._filter_chunks[idx % len(self._filter_chunks)])
        return chunk_i * self.chunk_size


class MesaBimanualAppleTrayOn60Dataset(MesaBimanualMultiTaskSingleDataset):
    """60-demo apple_tray_on slice of the multitask v2 zarr.

    Same data source and per-demo schema as the multitask baseline — just
    filtered to task_id=0. Used to settle whether the 0.5 plateau is caused by
    demo-scale (1 task × 60 demos also collapses) or by task diversity
    (multitask specifically triggers it).
    """
    task_filter = ("apple_tray_on",)
    train_copies = 100


class MesaBimanual2TaskDataset(RLBenchDataset):
    """Mesa bimanual dataset (2-task × 60-demo, depth-linearization fixed).

    Backed by a dedicated zarr that contains only `apple_tray_on` and
    `bottled_water_tray_on` with `task_id in {0, 1}` — no load-time filtering
    needed. Built by scripts/mesa/sbatch_build_2task_depth_fix.sh from the
    fk-mirror-repaired overfit HDF5s with the EXTENT=11.831 depth fix applied.
    """
    tasks = ("apple_tray_on", "bottled_water_tray_on")
    cameras = ("front", "wrist_right", "wrist_left")
    camera_inds = None
    camera_inds2d = None
    train_copies = 50
