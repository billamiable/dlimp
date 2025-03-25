import glob
import json
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow_datasets as tfds
from absl import logging
from dataset_builder import MultiThreadedDatasetBuilder
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pickle as pkl

# we ignore the small amount of data that contains >4 views
N_VIEWS = 4
IMAGE_SIZE = (480, 640)
DEPTH = 3
TRAIN_PROPORTION = 0.9

ORIG_NAMES = [f"images{i}" for i in range(N_VIEWS)]
NEW_NAMES = [f"image_{i}" for i in range(N_VIEWS)]

# start_image_id, end_image_id, gripper_min, gripper_max
# guideline for selecting start and end image id (id starts from 0):
#     1. gripper is inside view (at least partially seen)
#     2. aviod too large motion (normally after grasping, the motion will become large)
#            guideline 2 is actually no longer followed for pick up towel since large motion for xyz is within bridge dataset range
#            the biggest issue comes from roll,pitch,yaw large motion
valid_image_ids = {
    # pick up block
    "20250321_085908": [3, 29, 0.08130098134279251, 2.0923497676849365],
    "20250321_091158": [1, 24, 0.07669904083013535, 1.4097284078598022],
    "20250321_091510": [2, 23, 0.07976700365543365, 1.402058482170105],
    "20250321_091730": [1, 29, 0.06289321184158325, 1.4005244970321655],
    "20250321_091936": [2, 27, 0.0951068103313446, 1.418932318687439],
    "20250321_092140": [3, 26, 0.07363107800483704, 1.4127963781356812],
    "20250321_092359": [1, 21, 0.07516506314277649, 1.4296700954437256],
    "20250321_092553": [2, 22, 0.07976700365543365, 1.4388740062713623],
    "20250321_092800": [1, 22, 0.07516506314277649, 1.4127963781356812],
    "20250321_093013": [1, 24, 0.0782330259680748, 1.4173983335494995],
    # pick up towel
    "20250325_095001": [4, 27, -0.4939418137073517, 1.0354371070861816],
    "20250325_095309": [7, 29, -0.49087387323379517, 1.4173983335494995],
    "20250325_095443": [3, 26, -0.3850291967391968, 1.4419419765472412],
    # "20250325_095602": [11, 26, -0.4325825870037079, 1.4220002889633179], # skip for now due to too many times not seen gripper
    "20250325_095707": [5, 25, -0.42031073570251465, 1.4051264524459839], # once not seen TODO see if need to move to validation set
    "20250325_095802": [5, 26, -0.4832039475440979, 1.4434759616851807],
    "20250325_095900": [5, 30, -0.4279806613922119, 1.4035924673080444],
    "20250325_095956": [3, 26, -0.40957286953926086, 1.4434759616851807],
    "20250325_100113": [4, 32, -0.39576706290245056, 1.4127963781356812],
    "20250325_100213": [6, 30, -0.38963112235069275, 1.4358060359954834],
    "20250325_100350": [6, 28, -0.4279806613922119, 1.4035924673080444],
}


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        # depth should be uint16 (I;16), but PIL has a bug where it reads as int32 (I)
        # there are also few trajectories where it's uint8 (L) for some reason
        # we just cast to uint16 in both cases
        # test_img = np.asarray(im)
        # print(f"test_img: {test_img}")
        # print(f"test img size: {np.shape(test_img)}")
        # exit(1)
        assert im.mode == "RGB" or im.mode == "I" or im.mode == "L", (path, im.mode)
        assert im.size == (640, 480), (path, im.size)
        arr = np.array(im)
        if arr.ndim == 2:
            return arr[..., None].astype(np.uint16)
        else:
            assert arr.ndim == 3 and arr.shape[-1] == 3, (path, arr.shape)
            assert arr.dtype == np.uint8, (path, arr.dtype)
            return arr

    # you can speed things up significantly by skipping image decoding/re-encoding by using the line below,
    # but then you also need to skip the checks
    # return open(path, "rb").read()


'''
def process_images(path):  # processes images at a trajectory level
    image_dirs = set(os.listdir(str(path))).intersection(set(ORIG_NAMES))
    image_paths = [
        sorted(
            glob.glob(os.path.join(path, image_dir, "im_*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for image_dir in image_dirs
    ]

    filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
    assert all(x == filenames[0] for x in filenames), (path, filenames)

    d = {
        image_dir: [read_image(path) for path in p]
        for image_dir, p in zip(image_dirs, image_paths)
    }

    return d
'''

def process_images(path, use_valid_list=False):
    image_dict = {}
    folder_path = f"{path}/images"
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')), key=lambda x: os.path.basename(x))
    images = []
    for id, image_path in enumerate(image_paths):
        if use_valid_list:
            traj_name = path.split("/")[-1]
            # print(f"traj_name: {traj_name}")
            # exit(1)
            if traj_name not in valid_image_ids:
                    raise ValueError(f"no valid image ids available, please check your setup")
            start_image_id, end_image_id = valid_image_ids[traj_name][0], valid_image_ids[traj_name][1]
            if id < start_image_id or id > end_image_id:
                print(f"skip processing image index {id}")
                continue
        image = read_image(image_path)
        # print(f"test_img: {image}")
        # print(f"test img size: {np.shape(image)}")
        # exit(1)
        images.append(image)
    image_dict["images0"] = images
    return image_dict


def normalize_gripper(gripper):
    # TODO raw gripper value from 0.5 to 1.5
    normalized = gripper - 0.5
    if normalized < 0.5:
        return 0.0
    return 1.0


def matrix_to_xyz_rpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    rotation_matrix = matrix[:3, :3]

    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    state = [x, y, z, roll, pitch, yaw]
    return state

def process_pkl(path, use_valid_list=False):
    file_path = f"{path}/tfs.pkl"
    with open(file_path, 'rb') as file:
        print(f"File content read successfully from {file_path}.")
        tfs = pkl.load(file)

        states = []
        actions = []
        for id, item in enumerate(tfs):
            # # "20250319_124721.jpg" has same state as "20250319_124724.jpg", skip no movement frame
            # if id == 9:
            #     continue
            if use_valid_list:
                traj_name = path.split("/")[-1]
                if traj_name not in valid_image_ids:
                    raise ValueError(f"no valid image ids available, please check your setup")
                start_image_id, end_image_id = valid_image_ids[traj_name][0], valid_image_ids[traj_name][1]
                if id < start_image_id or id > end_image_id:
                    print(f"skip processing pkl data index {id}")
                    continue
            print(f"id: {id}")
            state_matrix, gripper = item[0], normalize_gripper(item[1])
            # print(f"state_matrix: {state_matrix}")
            # print(f"gripper: {gripper}")
            state = matrix_to_xyz_rpy(state_matrix)
            state.append(0.0)
            action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper])
            print(f"state: {state}, action: {action}")
            states.append(state)
            actions.append(action)

        return np.asarray(states), np.asarray(actions)

def get_action_from_state(states):
    if len(states) < 2:
        return []
    result = []
    for prev, curr in zip(states[:-1], states[1:]):
        new_list = [b - a for a, b in zip(prev, curr)]
        result.append(new_list)
    return result

def get_movement_action_statistics(states):
    # TODO large action needs different norm parameters
    movement_actions = get_action_from_state(states)
    for i, movement_action in enumerate(movement_actions):
        print(f"action: {i}, movement_action: {movement_action}")
    q01 = np.quantile(movement_actions, 0.01, axis=0).tolist()
    q99 = np.quantile(movement_actions, 0.99, axis=0).tolist()
    print(f"q01: {q01}")
    print(f"q99: {q99}")
    return movement_actions

def process_depth(path):
    depth_path = os.path.join(path, "depth_images0")
    if os.path.exists(depth_path):
        image_paths = sorted(
            glob.glob(os.path.join(depth_path, "im_*.png")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        return [read_image(path) for path in image_paths]
    else:
        return None


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"]


def process_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def process_lang(path):
    fp = os.path.join(path, "lang.txt")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()

    return text


class BridgeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for bridge dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "You can download the raw BridgeData from https://rail.eecs.berkeley.edu/datasets/bridge_release/data/."

    # temporary change, for testing
    NUM_WORKERS = 16
    CHUNKSIZE = 1000

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        # [VIP] using jpeg caused image precision loss
                                        encoding_format="png", # jpeg
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "depth_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (1,),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera depth observation (fixed position).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, consists of [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_depth_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if depth0 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        )


    '''
    @classmethod
    def _process_example(cls, example_input):
        """Process a single example."""
        path, camera_topics = example_input

        print(f"path: {path}, camera_topics: {camera_topics}")
        # exit(1)

        out = dict()

        out["images"] = process_images(path)
        test_images = out["images"]
        print(f"test_images： {test_images}")
        # exit(1)
        out["depth"] = process_depth(path)
        out["state"] = process_state(path)
        out["actions"] = process_actions(path)
        out["lang"] = process_lang(path)

        # temporary change
        out["lang"] = "put eggplant on plate"

        # data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
        date_time = datetime.strptime(path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")
        latency_shift = date_time < datetime(2021, 7, 23)

        # shift the actions according to camera latency
        if latency_shift:
            out["images"] = {k: v[1:] for k, v in out["images"].items()}
            out["state"] = out["state"][1:]
            out["actions"] = out["actions"][:-1]
            if out["depth"] is not None:
                out["depth"] = out["depth"][1:]

        # append a null action to the end
        out["actions"].append(np.zeros_like(out["actions"][0]))

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]["images0"])

        # assemble episode
        episode = []
        episode_metadata = dict()

        # map original image name to correct image name according to logged camera topics
        orig_to_new = dict()
        for image_idx in range(len(out["images"])):
            orig_key = ORIG_NAMES[image_idx]

            if camera_topics[image_idx] in [
                "/cam0/image_raw",
                "/camera0/color/image_raw",
                "/D435/color/image_raw",
            ]:
                # fixed cam should always be image_0
                new_key = "image_0"
                # assert new_key[-1] == orig_key[-1], episode_path
            elif camera_topics[image_idx] == "/wrist/image_raw":
                # wrist cam should always be image_3
                new_key = "image_3"
            elif camera_topics[image_idx] in [
                "/cam1/image_raw",
                "/cam2/image_raw",
                "/cam3/image_raw",
                "/cam4/image_raw",
                "/camera1/color/image_raw",
                "/camera3/color/image_raw",
                "/camera2/color/image_raw",
                "/camera4/color/image_raw",
                "/blue/image_raw",
                "/yellow/image_raw",
            ]:
                # other cams can be either image_1 or image_2
                if "image_1" in list(orig_to_new.values()):
                    new_key = "image_2"
                else:
                    new_key = "image_1"
            else:
                raise ValueError(f"Unexpected camera topic {camera_topics[image_idx]}")

            orig_to_new[orig_key] = new_key
            episode_metadata[f"has_{new_key}"] = True

        # record which images are missing
        missing_keys = set(NEW_NAMES) - set(orig_to_new.values())
        for missing in missing_keys:
            episode_metadata[f"has_{missing}"] = False

        episode_metadata["has_depth_0"] = out["depth"] is not None

        instruction = out["lang"]
        print(f"instruction: {instruction}")
        # action_size = len(out["actions"])
        # img_size = len(out["images"]["images0"])
        # print(f"action size: {action_size}, img size: {img_size}")
        # exit(1)

        # example_state = out["state"][0]
        # print(f"example_state: {example_state}")
        # exit(1)

        for i in range(len(out["actions"])):
            observation = {
                "state": out["state"][i].astype(np.float32),
            }

            for orig_key in out["images"]:
                new_key = orig_to_new[orig_key]
                observation[new_key] = out["images"][orig_key][i]
            for missing in missing_keys:
                observation[missing] = np.zeros(IMAGE_SIZE + (3,), dtype=np.uint8)
            if episode_metadata["has_depth_0"]:
                observation["depth_0"] = out["depth"][i]
            else:
                observation["depth_0"] = np.zeros(IMAGE_SIZE + (1,), dtype=np.uint16)

            episode.append(
                {
                    "observation": observation,
                    "action": out["actions"][i].astype(np.float32),
                    "is_first": i == 0,
                    "is_last": i == (len(out["actions"]) - 1),
                    "language_instruction": instruction,
                }
            )

        episode_metadata["file_path"] = path
        episode_metadata["has_language"] = bool(instruction)

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        # use episode path as key
        return path, sample

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # each path is a directory that contains dated directories
        paths = glob.glob(os.path.join(dl_manager.manual_dir, *("*" * (DEPTH - 1))))

        print(f"paths: {paths}")
        # exit(1)

        train_inputs, val_inputs = [], []

        for path in paths:
            for dated_folder in os.listdir(path):
                # a mystery left by the greats of the past
                if "lmdb" in dated_folder:
                    continue

                search_path = os.path.join(
                    path, dated_folder, "raw", "traj_group*", "traj*"
                )
                all_traj = glob.glob(search_path)
                # enforce order
                all_traj = sorted(all_traj)

                if not all_traj:
                    print(f"no trajs found in {search_path}")
                    continue

                config_path = os.path.join(path, dated_folder, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                    camera_topics = config["agent"]["env"][1]["camera_topics"]
                    print(f"camera_topics: {camera_topics}")
                    # exit(1)
                else:
                    # assumed camera topics if no config.json exists
                    camera_topics = [
                        "/D435/color/image_raw",
                        "/blue/image_raw",
                        "/yellow/image_raw",
                        "/wrist/image_raw",
                    ]
                all_inputs = [(t, camera_topics) for t in all_traj]

                train_inputs += all_inputs[: int(len(all_inputs) * TRAIN_PROPORTION)]
                val_inputs += all_inputs[int(len(all_inputs) * TRAIN_PROPORTION) :]

        print(f"all_inputs: {all_inputs}")
        print(f"train_inputs: {train_inputs}")
        print(f"val_inputs: {val_inputs}")
        # exit(1)

        logging.info(
            "Converting %d training and %d validation files.",
            len(train_inputs),
            len(val_inputs),
        )
        return {
            "train": iter(train_inputs),
            "val": iter(val_inputs),
        }
    '''





    @classmethod
    def _process_example(cls, example_input):
        """Process a single example."""
        path = example_input
        print(f"path: {path}")
        # exit(1)

        out = dict()

        # temporary change for use_valid_list=False
        out["images"] = process_images(path, use_valid_list=True)
        out["depth"] = None
        out["state"], out["actions"] =  process_pkl(path, use_valid_list=True)
        # test_images = out["images"]
        # test_state = out["state"]
        # test_action = out["actions"]
        # print(f"test_images： {test_images}")
        # print(f"test_state: {test_state}, test_action: {test_action}")
        # get_movement_action_statistics(out["state"])
        # exit(1)

        # temporary change
        # out["lang"] = "pick up banana"
        # TODO check if need to use "pick up red block"
        # out["lang"] = "pick up block"
        out["lang"] = "pick up towel"

        # temporary change
        # img_num = len(out["images"]["images0"])
        # out["actions"] = out["actions"][:img_num]
        # out["state"] = out["state"][:img_num]

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]["images0"])

        print("successfully pass assert")
        # exit(1)

        # assemble episode
        episode = []
        episode_metadata = dict()

        # map original image name to correct image name
        orig_to_new = dict()
        orig_key = "images0"
        new_key = "image_0"
        orig_to_new[orig_key] = new_key
        episode_metadata[f"has_{new_key}"] = True

        # record which images are missing
        missing_keys = set(NEW_NAMES) - set(orig_to_new.values())
        for missing in missing_keys:
            episode_metadata[f"has_{missing}"] = False
        # print(f"missing_keys: {missing_keys}")
        # exit(1)

        episode_metadata["has_depth_0"] = out["depth"] is not None

        instruction = out["lang"]
        print(f"instruction: {instruction}")
        # action_size = len(out["actions"])
        # img_size = len(out["images"]["images0"])
        # print(f"action size: {action_size}, img size: {img_size}")
        # exit(1)

        # test_state = []
        for i in range(len(out["actions"])):
            # temporary change for pick up banana, for pick up block, will use valid_image_ids instead
            # # "20250319_124721.jpg" has same state as "20250319_124724.jpg", skip no movement frame
            # if i == 9:
            #     print(f"skip frame due to static movement")
            #     continue
            observation = {
                "state": out["state"][i].astype(np.float32),
            }
            # test_state.append(out["state"][i].astype(np.float32))

            # test_image = out["images"]["images0"][0]
            # print(f"test_img: {test_image}")
            # print(f"test img size: {np.shape(test_image)}")
            # exit(1)

            for orig_key in out["images"]:
                new_key = orig_to_new[orig_key]
                observation[new_key] = out["images"][orig_key][i]
            for missing in missing_keys:
                observation[missing] = np.zeros(IMAGE_SIZE + (3,), dtype=np.uint8)
            if episode_metadata["has_depth_0"]:
                observation["depth_0"] = out["depth"][i]
            else:
                observation["depth_0"] = np.zeros(IMAGE_SIZE + (1,), dtype=np.uint16)

            episode.append(
                {
                    "observation": observation,
                    "action": out["actions"][i].astype(np.float32),
                    "is_first": i == 0,
                    "is_last": i == (len(out["actions"]) - 1),
                    "language_instruction": instruction,
                }
            )

        # get_movement_action_statistics(test_state)
        # exit(1)

        episode_metadata["file_path"] = path
        episode_metadata["has_language"] = bool(instruction)

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        # use episode path as key
        return path, sample


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        train_inputs, val_inputs = [], []


        all_tasks = glob.glob(os.path.join(dl_manager.manual_dir, '*', ''))
        all_inputs = []
        for folder in all_tasks:
            all_trajs = glob.glob(os.path.join(folder, '*', ''))
            # remove the ending "/"
            all_trajs = [traj.rstrip('/') for traj in all_trajs]
            all_inputs.extend(all_trajs)
        all_inputs = sorted(all_inputs)

        train_inputs += all_inputs[: int(len(all_inputs) * TRAIN_PROPORTION)]
        val_inputs += all_inputs[int(len(all_inputs) * TRAIN_PROPORTION) :]

        print(f"all_inputs: {all_inputs}")
        print(f"train_inputs: {train_inputs}")
        print(f"val_inputs: {val_inputs}")
        # exit(1)

        logging.info(
            "Converting %d training and %d validation files.",
            len(train_inputs),
            len(val_inputs),
        )
        return {
            "train": iter(train_inputs),
            "val": iter(val_inputs),
        }