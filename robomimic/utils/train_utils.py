"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

from robomimic.classifier.classifier import MultiTrajectoryDataset

import gc
from memory_profiler import profile
import sys
from memory_profiler import LogFile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)


def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute
    )
    dataset = SequenceDataset(**ds_kwargs)

    return dataset

def concatenate_state_dict(state_dict):
    # Initialize an empty list to hold all the arrays
    state_list = []
    
    # Iterate over all key-value pairs in the dictionary
    for key, value in state_dict.items():
        # Ensure the value is a numpy array and then flatten it before adding to the list
        if isinstance(value, np.ndarray):
            state_list.append(value.flatten())
    
    # Concatenate all arrays in the list into a single numpy array
    state_array = np.concatenate(state_list)
    
    return state_array

def run_rollout(
        policy, 
        env, 
        horizon,
        epoch,
        episode,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        classifier=None,
        video_path=None,
        at_end = False
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)


    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics

    rollout = []
    actions = []
    saved_frames = []

    num_steps = 0

    state_indices_list = {'lift': [(0, 10), (37, 40), (40, 44), (51, 53)], 'square': [(0,14), (41, 44), (44, 48), (55, 57)],
    'PickPlaceCan': [(0,14), (41, 44), (44, 48), (55, 57)] }

    state_indices = state_indices_list[env.name]

    classifier.eval()

    # For Classifier Integration train the classifier here with a frozen policy
    try:
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # print("OB_dict: ", ob_dict)

            state = concatenate_state_dict(ob_dict)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None and at_end:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    # video_writer.append_data(video_img)
                    # update_frame(step_i, video_img, classifier, video_writer)
                    saved_frames.append(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break
            
            state = np.hstack([state[start:end] for start, end in state_indices])
            rollout.append(state)
            actions.append(ac)

            num_steps += 1

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    trajectory_lengths = [len(rollout)]
    result = [success["task"]]
    single_trajectory_dataset = MultiTrajectoryDataset(rollout, actions, result, trajectory_lengths, classifier.num_past, classifier.num_future,'train')

    true_labels, predicted_labels = evaluate_trajectory_performance(classifier, single_trajectory_dataset, 0, classifier.threshold, saved_frames, video_path, at_end, device='cpu')
    # with h5py.File(f'bc_trajectories_manipulator_transport.hdf5', 'a') as f:
    #     demo_group = f.create_group(f"data/demo_{epoch}_{episode}")
    #     actions_ds = demo_group.create_dataset("actions", (num_steps, 14), dtype='f8')
    #     states_ds = demo_group.create_dataset("states", (num_steps, 131), dtype='f8')
        

    #     actions_ds[:] = np.array(actions)
    #     states_ds[:] = np.array(rollout)

    #     demo_group.attrs['label'] = success["task"]


    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])

    # Classifier Stuff

    results["classifier_true_labels"] = true_labels
    results["classifier_predicted_labels"] = predicted_labels

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results, rollout, actions, success["task"]

def evaluate_trajectory_performance(classifier, dataset, trajectory_idx, threshold, saved_frames, video_path, at_end, device='cpu'):
    classifier.eval()
    classifier.to(device)

    # Filter trajectory indices for the specified trajectory
    trajectory_indices = [
        idx for idx, (traj_idx, _) in enumerate(dataset.indices) if traj_idx == trajectory_idx
    ]

    if at_end:
        # Get video properties
        frame_width = 512
        frame_height = 512
        fps = 20
        total_frames = len(saved_frames)

        # Debug: print video properties
        print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total Frames: {total_frames}, Path: {video_path}")

        if frame_width == 0 or frame_height == 0 or fps == 0:
            print("Invalid video properties.")
            return 0.0  # Invalid video properties

        # Set up video writer with the 'mp4v' codec for MP4 compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Switch to 'mp4v' codec for MP4 files

        combined_video_writer = imageio.get_writer(video_path, fps=fps)
        
        # Set up plots
        fig_scatter, ax_scatter = plt.subplots()  # Scatter plot figure
        fig_logit, ax_logit = plt.subplots()      # Logit line graph figure

    # Lists to store plot data
    x_coords, y_coords, colors, logits, predicted_labels, true_labels, indices = [], [], [], [], [], [], []
    logit_values, time_points = [], []

    # Initialize variables for performance evaluation
    total_correct_predictions = 0
    total_predictions = 0



    def update_frame(i):
        # Read the next frame from the video
        correct_predictions = 0
        num_predictions = 0
        frame = saved_frames[i]

        # Get the state-action sequence and true label for the current index
        points_per_frame = 5

        for k in range(points_per_frame):
            idx = points_per_frame * i + k
            if (idx < len(trajectory_indices)):
                state_action_seq, true_label = dataset[trajectory_indices[idx]]
                state_action_seq = state_action_seq.to(device)


                true_label = true_label.to(device)

                # Predict the label using the model
                output = classifier(state_action_seq.unsqueeze(0))  # Add batch dimension
                logit = output.squeeze().item()
                predicted_label = (output > threshold).float()

                predicted_labels.append(predicted_label.item())
                logits.append(logit)
                true_labels.append(true_label.item())
                indices.append(idx)

                correct_predictions += 1 if predicted_label.item() == true_label.item() else 0
                num_predictions += 1
                color = 'green' if predicted_label.item() == true_label.item() else 'red'
                colors.append(color)
            
            else:
                return False, correct_predictions, num_predictions, true_labels, predicted_labels

        # Update scatter plot
        ax_scatter.cla()
        ax_scatter.set_xlabel('Time (frames)')
        ax_scatter.set_ylabel('Classification')
        ax_scatter.set_title('Real-Time Model Performance Over Time')

        time_axis = list(range(1, len(predicted_labels) + 1))

        # Scatter plot using time as the x-axis, with the same colors for success/failure
        ax_scatter.scatter(time_axis, [1] * len(time_axis), c=colors, marker='o', edgecolor='k')

        for i in range(len(time_axis)):
            t = time_axis[i]  # x-coordinate on time axis
            y = 1  # y-coordinate since you are plotting at y=1
            ax_scatter.annotate(f"{indices[i]} (P: {int(predicted_labels[i])}, T: {int(true_labels[i])}, Logit: {logits[i]:.2f})",
                                (t, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='blue')

        # Render scatter plot to a buffer
        fig_scatter.canvas.draw()
        scatter_image = np.frombuffer(fig_scatter.canvas.tostring_rgb(), dtype=np.uint8)
        scatter_image = scatter_image.reshape(fig_scatter.canvas.get_width_height()[::-1] + (3,))

        # Resize the scatter plot image to match the video frame size
        scatter_image_resized = cv2.resize(scatter_image, (frame_width, frame_height))

        # Concatenate the video frame and the scatter plot image side by side
        scatter_combined_frame = np.concatenate((frame, scatter_image_resized), axis=1)

        # Write the combined frame to the scatter plot video
        # scatter_video_writer.write(scatter_combined_frame)

        # Update logit line plot
        ax_logit.cla()
        ax_logit.set_ylim(0, 1)
        ax_logit.set_xlabel('Time (frames)')
        ax_logit.set_ylabel('Probability Value')
        ax_logit.set_title('Probability Progression Over Time')

        # Plot logit values over time
        ax_logit.plot(indices, logits, label='Probability', color='blue')
        ax_logit.legend()

        # Render logit plot to a buffer
        fig_logit.canvas.draw()
        logit_image = np.frombuffer(fig_logit.canvas.tostring_rgb(), dtype=np.uint8)
        logit_image = logit_image.reshape(fig_logit.canvas.get_width_height()[::-1] + (3,))

        # Resize the logit plot image to match the video frame size
        logit_image_resized = cv2.resize(logit_image, (frame_width, frame_height))

        # Concatenate the video frame and the logit plot image side by side
        logit_combined_frame = np.concatenate((frame, logit_image_resized), axis=1)

        # Write the combined frame to the logit video
        # logit_video_writer.write(logit_combined_frame)

        combined_frame = np.concatenate((frame, scatter_image_resized, logit_image_resized), axis=1)

        # Write the combined frame to the third video
        combined_video_writer.append_data(combined_frame)


        return True, correct_predictions, num_predictions, true_labels, predicted_labels
    
    if at_end:
        frame_idx = 0
        while frame_idx < len(saved_frames):
            update, correct_predictions, num_predictions, true_labels, predicted_labels = update_frame(frame_idx)
            total_correct_predictions += correct_predictions
            total_predictions += num_predictions
            if not update:
                break
            frame_idx += 1
        
        combined_video_writer.close()
        plt.close(fig_scatter)
        plt.close(fig_logit)
    else:
        for i in range(len(trajectory_indices)):
            state_action_seq, true_label = dataset[trajectory_indices[i]]
            state_action_seq = state_action_seq.to(device)


            true_label = true_label.to(device)

            # Predict the label using the model
            output = classifier(state_action_seq.unsqueeze(0))  # Add batch dimension
            logit = output.squeeze().item()
            predicted_label = (output > threshold).float()

            predicted_labels.append(predicted_label.item())
            logits.append(logit)
            true_labels.append(true_label.item())
            indices.append(i)

            total_correct_predictions += 1 if predicted_label.item() == true_label.item() else 0
            total_predictions += 1
        


    return true_labels, predicted_labels


def rollout_with_stats(
        policy,
        classifier,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)

    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0

        all_states = []
        all_actions = []
        all_results = []
        trajectory_lengths = []
        
        true_labels = []
        predicted_labels = []

        for ep_i in iterator:

            env_video_writer = None
            if write_video:
                video_str = "_epoch_{}_episode_{}.mp4".format(epoch, ep_i) if epoch is not None else "_episode_{}.mp4".format(ep_i)
                video_path = os.path.join(video_dir, "{}{}".format(env_name, video_str))
                print("video writes to " + video_path)
                env_video_writer = imageio.get_writer(video_path, fps=20)
            rollout_timestamp = time.time()
            at_end = (ep_i == num_episodes - 1)
            rollout_info, states, actions, result = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                epoch = epoch,
                episode = ep_i,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
                classifier=classifier,
                video_path=video_path,
                at_end = at_end
            )

            # Classifier Stuff

            all_states.extend(states)
            all_actions.extend(actions)
            all_results.append(result)
            trajectory_lengths.append(len(states))

            rollout_info["time"] = time.time() - rollout_timestamp

            true_labels.extend(rollout_info["classifier_true_labels"])
            predicted_labels.extend(rollout_info["classifier_predicted_labels"])

            rollout_info.pop("classifier_true_labels")
            rollout_info.pop("classifier_predicted_labels")

            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))


            gc.collect()
        
        # Train Classifier using the validation rollouts

        dataset = MultiTrajectoryDataset(all_states, all_actions, all_results, trajectory_lengths, classifier.num_past, classifier.num_future, 'train')

        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        num_epochs = 3

        if (not classifier.checkpoint):
            train_classifier(classifier, num_epochs, data_loader)
        else:
            print("Using checkpointed Classifier Not Training")

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        
        # Classifier Precision, Recall, F1, Accuracy
        rollout_logs_mean["Classifier Precision"] = precision_score(true_labels, predicted_labels)
        rollout_logs_mean["Classifier Recall"] = recall_score(true_labels, predicted_labels)
        rollout_logs_mean["Classifier F1"] = f1_score(true_labels, predicted_labels)
        rollout_logs_mean["Classifier Accuracy"] = accuracy_score(true_labels, predicted_labels)

        all_rollout_logs[env_name] = rollout_logs_mean

    # if video_path is not None:
    #     # close video writer that was used for all envs
    #     video_writer.close()

    return all_rollout_logs, video_paths

def train_classifier(classifier, num_epochs, data_loader):

    criterion = nn.BCELoss() 
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_loader_tqdm = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for state_action_seq, labels in train_loader_tqdm:

            # Forward pass
            outputs = classifier(state_action_seq)
            if outputs.dim() == 1 and outputs.size(0) == 1:
                # Don't squeeze the output if it's already a scalar
                outputs_squeezed = outputs
            else:
                outputs_squeezed = outputs.squeeze()

            if outputs_squeezed.dim() == 0:  # If output is scalar
                outputs_squeezed = outputs_squeezed.unsqueeze(0)  # Make it a 1D tensor
            if labels.dim() == 0:  # If label is scalar
                labels = labels.unsqueeze(0)  # Make it a 1D tensor

            loss = criterion(outputs_squeezed, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()

            predictions = (outputs.squeeze() > classifier.threshold).float()

            # Calculate the number of correct predictions
            train_correct_predictions += (predictions == labels).sum().item()
            train_total_predictions += labels.size(0)
            
            # Update TQDM bar
            train_loader_tqdm.set_postfix({"Train Loss": loss.item()})

        avg_train_loss = train_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Accuracy {train_correct_predictions/train_total_predictions}')
        


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
