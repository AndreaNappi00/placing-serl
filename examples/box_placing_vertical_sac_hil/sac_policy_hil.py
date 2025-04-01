#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from datetime import datetime
import os

import gymnasium as gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import TransformReward

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.data.data_store import populate_data_store

from serl_launcher.wrappers.chunking import ChunkingWrapper
from ur_env.envs.relative_env import RelativeFrame, RelativeRewardVertical

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages, ScaleObservationWrapper, SerlObsWrapperTrajBox
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2rotvecWrapper

import ur_env

from serl_launcher.utils.sampling_utils import TemporalActionEnsemble


from colorama import Fore, Style

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_basic_env", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", "sac_drq_policy", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 150, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 8, "UTD ratio.")
flags.DEFINE_integer("reward_scale", 1, "Reward Scale to help out SAC algorithm")

flags.DEFINE_integer("max_steps", 100000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")
flags.DEFINE_multi_string("demo_paths", None,
                          "paths to demos")

flags.DEFINE_integer("random_steps", 10, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 10, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 1000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 3, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 1000, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", '/home/andrea/Code/placing-serl/examples/box_placing_vertical_sac_hil/checkpoints',
                    "Path to save checkpoints.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step")
flags.DEFINE_string("eval_checkpoint_path", None, "evaluate the policy from ckpt from this path")

flags.DEFINE_string("log_rlds_path", '/home/andrea/Code/placing-serl/examples/box_placing_vertical_sac_hil/rlds',
                    "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("load_checkpoint_path", '/home/andrea/Code/placing-serl/examples/box_placing_vertical_sac_hil/checkpoints',
                    "Path to load previously saved checkpoints and start training from them.")
flags.DEFINE_boolean("evaluation", False, "Evaluation mode.")

flags.DEFINE_string("wandb_project", "placing", "Wandb project name.")

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))
def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

test_bc = False

##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng, wandb_logger=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step and FLAGS.evaluation:
        success_counter = 0
        time_list = []
        running_reward = 0.0 

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.eval_checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        step = 0
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))
                # print(actions)

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                running_reward += reward
                wandb_logger.log(info, step=step)
                
                forces_status = f"{Fore.GREEN if info.get('forces') else Fore.RED}{'True' if info.get('forces') else 'False'}{Style.RESET_ALL}"
                box_status = f"{Fore.GREEN if info.get('box_pose') else Fore.RED}{'True' if info.get('box_pose') else 'False'}{Style.RESET_ALL}"
                print(f"Forces: {forces_status}, Box: {box_status}, Reward: {reward}, Running Reward: {running_reward}", end='\r')
                
                step += 1

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print("execution time: ", dt)

                    success_counter += int(reward > 50)
                    print("reward: ", reward)
                    print("running reward: ", running_reward)
                    print(f"{success_counter}/{episode + 1}")
                    running_reward = 0.0

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    print(f"obs:  {obs}")
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    consecutive_successes = 0
    success_counter = 0
    
    action_ensemble = TemporalActionEnsemble(activated=False)
    pbar = tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")
        action_ensemble.reset()

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                # print("sampling randomly!")
                actions = env.action_space.sample()

            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                    # deterministic=False,              # sample without argmax for more diverse actions
                )
                actions = np.asarray(jax.device_get(actions))
            ensembled_action = action_ensemble.sample(actions)

        # Step environment
        with timer.context("step_env"):
            if test_bc:
                next_obs, reward, done, truncated, info = env.step(ensembled_action)
            else:
                next_obs, reward, done, truncated, info = env.step(actions)
                
            # override the action with the intervention action
            if "hil_action" in info:
                actions = info.pop("hil_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
                
            wandb_logger.log(info, step=step)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            
            forces_status = f"{Fore.GREEN if info.get('forces') else Fore.RED}{'True' if info.get('forces') else 'False'}{Style.RESET_ALL}"
            box_status = f"{Fore.GREEN if info.get('box_pose') else Fore.RED}{'True' if info.get('box_pose') else 'False'}{Style.RESET_ALL}"
            pbar.set_description(f"Forces: {forces_status}, Box: {box_status}")

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                compare_success_count = (success_counter + 1 == env.unwrapped.config.SUCCESS_COUNT) # true if there was not a success, otherwise false
                success_counter = env.unwrapped.config.SUCCESS_COUNT
                consecutive_successes = (consecutive_successes + 1 if compare_success_count else 0)
                info["success_counter"] = success_counter
                info["consecutive_successes"] = consecutive_successes
                
                print(f"running return: {running_return}")
                info = np.asarray(info)
                stats = {"train": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0 and step:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()
    
    
    update_step = 0
    if FLAGS.eval_checkpoint_step:
        print_yellow("loading checkpoint")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.load_checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        server.publish_network(agent.state.params)
        print_yellow(f"sent network to actor from checkpoint {FLAGS.eval_checkpoint_step}")

    else:
        server.publish_network(agent.state.params)
        print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    try:
        for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
            # Train the networks
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train"):
                if FLAGS.utd_ratio == 1:
                    agent, update_info = agent.update(batch=batch)  # try it without utd
                else:
                    agent, update_info = agent.update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)
                agent = jax.block_until_ready(agent)

                # publish the updated network
                server.publish_network(agent.state.params)

            if update_steps % FLAGS.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=update_steps)
                wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
                wandb_logger.log({"replay_buffer_size": len(replay_buffer)})

            if FLAGS.checkpoint_period and (update_steps + 1) % FLAGS.checkpoint_period == 0:
                assert FLAGS.checkpoint_path is not None
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=update_steps + 1, keep=100
                )

            update_steps += 1
    finally:
        print("closing learner, clearning up...")
        del replay_buffer


##############################################################################


def main(_):
    
    assert FLAGS.batch_size % num_devices == 0
    FLAGS.checkpoint_path = FLAGS.checkpoint_path + " " + datetime.now().strftime("%m%d-%H:%M")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        fake_env=FLAGS.learner,
        max_episode_length=FLAGS.max_traj_length,
        camera_mode="none",
    )
    if FLAGS.actor:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = RelativeRewardVertical(env)
    env = Quat2rotvecWrapper(env)
    env = ScaleObservationWrapper(env)
    env = SerlObsWrapperTrajBox(env)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    # env = TransformReward(env, lambda r: FLAGS.reward_scale * r)
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    print(f"obs shape: {env.observation_space.sample().shape}")
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            type="replay_buffer",
            rlds_logger_path=FLAGS.log_rlds_path,
            preload_rlds_path=FLAGS.preload_rlds_path,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="placing",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger
    
    wandb_logger = make_wandb_logger(
            project="placing",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
    
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            type="replay_buffer",
            rlds_logger_path=FLAGS.log_rlds_path,
            preload_rlds_path=FLAGS.preload_rlds_path,
        )

        if FLAGS.preload_rlds_path is None and FLAGS.demo_paths is not None:
            print(f"loaded demos from {FLAGS.demo_paths}")  # load demo trajectories the old way
            replay_buffer = populate_data_store(replay_buffer, FLAGS.demo_paths, reward_scaling=FLAGS.reward_scale)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        try:
            actor(agent, data_store, env, sampling_rng, wandb_logger)
            print_green("actor loop finished")
        except KeyboardInterrupt:
            print_green("actor loop interrupted")
        finally:
            env.close()

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
