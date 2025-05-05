from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import quaternion

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

class RLBenchO1Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, qpos or RTX version: consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, qpos or RTX version: consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_reason': tfds.features.Text(
                        doc='Language Reason.'
                    ),
                    'is_perturb': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Whether this step is from a perturbed trajectory.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/gpfs/ebd/runs_vla_data/train'),
            'val': self._generate_examples(path='/gpfs/ebd/runs_vla_data/val'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        error_count = 0
        error_log_path = f"errors_{path.replace('/', '_')}.txt"
        
        def _get_episode_paths(path):
            episode_paths = []
            tasks = os.listdir(path)
            for task in tasks:
                task_path = os.path.join(path, task)
                if not os.path.isdir(task_path):
                    continue
                episodes = os.listdir(task_path)
                for episode in episodes:
                    episode_path = os.path.join(task_path, episode, "video")
                    if not os.path.isdir(episode_path):
                        continue
                    episode_paths.append(episode_path)
            return episode_paths
        
        def _process_pose_to_state(pose_dict):
            """Convert pose dictionary to 8-dimensional state vector
            
            Args:
                pose_dict: Dictionary containing 'pos', 'ori', and 'gripper_open'
                
            Returns:
                np.array: 8-dimensional state vector [pos(3) + ori(4) + gripper(1)]
            """
            # Extract position (3D) and orientation (4D quaternion)
            pos = np.array(pose_dict['pos'], dtype=np.float32)  # [x, y, z]
            ori = np.array(pose_dict['ori'], dtype=np.float32)  # [qx, qy, qz, qw]
            
            # Convert gripper state to float (1D)
            gripper = 0.0 if pose_dict['gripper_open'] else 1.0
            
            # Concatenate into 8D vector
            state = np.concatenate([pos, ori, [gripper]], dtype=np.float32)
            return state
        
        def _get_relative_pose(pose1, pose2):
            trans_1, trans_2 = pose1[:3], pose2[:3]
            trans_diff = trans_2 - trans_1
            rot_1, rot_2 = pose1[3:7], pose2[3:7]  # [x, y, z, w]
            rot_1 = quaternion.from_float_array([rot_1[3], rot_1[0], rot_1[1], rot_1[2]])  # Convert to [w, x, y, z]
            rot_2 = quaternion.from_float_array([rot_2[3], rot_2[0], rot_2[1], rot_2[2]])
            rot_diff = rot_2 * rot_1.inverse()
            rot_diff = -rot_diff if rot_diff.w < 0 else rot_diff
            
            rot_mat = R.from_quat(quaternion.as_float_array(rot_diff), scalar_first=True)
            euler_diff = rot_mat.as_euler('xyz', degrees=False)
            
            # # can recon rot_2 from rot_1 and euler_diff as follows:
            # rot_diff_recon = R.from_euler('xyz', euler_diff, degrees=False).as_quat()  # [x, y, z, w]
            # rot_diff_recon = quaternion.from_float_array([rot_diff_recon[3], rot_diff_recon[0], rot_diff_recon[1], rot_diff_recon[2]])  # Convert to [w, x, y, z]
            # rot_diff_recon = -rot_diff_recon if rot_diff_recon.w < 0 else rot_diff_recon
            # rot_2_recon = rot_diff_recon * rot_1  # still in [w, x, y, z], need to convert to [x, y, z, w]
            
            return np.concatenate([trans_diff, euler_diff, pose2[7:8]], dtype=np.float32)
    
        def _parse_example(episode_path):
            nonlocal error_count
            data = []
            
            for subdir in os.listdir(episode_path):
                try:
                    subdir_path = os.path.join(episode_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                        
                    # Get image paths rather than loading them
                    if 'expert' in subdir:
                        rgb_path = os.path.join(subdir_path, 'front_rgb', 'begin.png')
                        depth_path = os.path.join(subdir_path, 'front_depth', 'begin.png')
                        info_path = os.path.join(subdir_path, 'info.json')
                        
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        curr_pose = _process_pose_to_state(info['prev_pose'])
                        next_pose = _process_pose_to_state(info['current_pose'])
                        
                        sample = {
                            'observation': {
                                'image': rgb_path,  # Just store the path
                                'depth_image': depth_path,  # Just store the path
                                'state': curr_pose
                            },
                            'action': _get_relative_pose(curr_pose, next_pose),
                            'language_instruction': info['lang_goal'],
                            'language_reason': f"To achieve the goal, the robot should now {info['subgoal'][11:]}",
                            'is_perturb':False
                        }
                        
                    elif 'perturb' in subdir:
                        
                        rgb_path = os.path.join(subdir_path, 'front_rgb', 'end.png')
                        depth_path = os.path.join(subdir_path, 'front_depth', 'end.png')
                        info_path = os.path.join(subdir_path, 'info.json')
                        
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        curr_pose = _process_pose_to_state(info['current_pose'])
                        next_pose = _process_pose_to_state(info['correct_pose'])
                            
                        sample = {
                            'observation': {
                                'image': rgb_path,  # Just store the path
                                'depth_image': depth_path,  # Just store the path
                                'state': curr_pose
                            },
                            'action': _get_relative_pose(curr_pose, next_pose),
                            'language_instruction': info['lang_goal'],
                            'language_reason': f"This step failed because {info['failure_reason_gpt']}. To correct this, the robot needs to {info['correction_instruction_gpt']}",
                            'is_perturb':True
                        }
                    else:
                        raise ValueError(f"Unknown subdir: {subdir}")
                        
                    if 'expert' in subdir or 'perturb' in subdir:
                        data.append(sample)
                        
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing episode {episode_path}, subdir {subdir}: {str(e)}\n"
                    # print(error_msg, end='')
                    with open(error_log_path, 'a') as f:
                        f.write(error_msg)
                    continue
            
            # create output data sample
            samples = {
                'steps': data,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
                    
            return episode_path, samples

        episode_paths = _get_episode_paths(path)
        
        # Clear existing log file
        open(error_log_path, 'w').close()
        
        # Add tqdm progress bar
        for episode_path in tqdm(episode_paths, desc=f"Processing {path}", unit="episode"):
            yield _parse_example(episode_path)
            
        print(f"\nTotal errors encountered: {error_count}")
        print(f"Error details saved to: {error_log_path}")

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


