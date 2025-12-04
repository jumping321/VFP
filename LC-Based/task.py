"""Vision-guided flight task."""
# ruff: noqa: F821

from typing import Optional
import numpy as np

from dm_control.utils import rewards
from dm_control.composer.observation import observable
from dm_control import composer
from flybody1.tasks.arenas.hills import SineTrench

from flybody1.tasks.pattern_generators import (WingBeatPatternGenerator)
from flybody1.tasks.task_utils import neg_quat
from flybody1.tasks.base import Flying
    
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import os

class VisionFlightImitationWBPG(Flying):
    """Vision-based flight with controllable Wing Beat Pattern Generator."""

    def __init__(self,
                 wbpg: WingBeatPatternGenerator,
                 floor_contacts_fatal: bool = True,
                 eye_camera_fovy: float = 150.,
                 eye_camera_size: int = 32,
                 target_height_range: tuple = (0.5, 0.8),
                 target_speed_range: tuple = (20, 40),
                 init_pos_x_range: Optional[tuple] = (-5, -5),
                 init_pos_y_range: Optional[tuple] = (0, 0),
                 **kwargs):
        """Task of learning a policy for flying and maneuvering while using a
            wing beat pattern generator with controllable wing beat frequency.

        Args:
            wpg: Wing beat generator.
            floor_contacts_fatal: Whether to terminate the episode when the fly
                contacts the floor.
            eye_camera_fovy: Field of view of the eye camera.
            eye_camera_size: Size of the eye camera.
            target_height_range: Range of target height.
            target_speed_range: Range of target speed.
            init_pos_x_range: Range of initial x position.
            init_pos_y_range: Range of initial y position.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False,
                         num_user_actions=1,
                         eye_camera_fovy=eye_camera_fovy,
                         eye_camera_size=eye_camera_size,
                         **kwargs)
        self._wbpg = wbpg
        self._floor_contacts_fatal = floor_contacts_fatal
        self._eye_camera_size = eye_camera_size
        self._target_height_range = target_height_range
        self._target_speed_range = target_speed_range
        self._init_pos_x_range = init_pos_x_range
        self._init_pos_y_range = init_pos_y_range
        # 初始化左右眼处理器，各自独立状态
        self._lmc_left = LMCsProcessor(time_step=1,save_dir="left")
        self._lmc_right = LMCsProcessor(time_step=1,save_dir="right")
        
        # Remove all light.
        for light in self._walker.mjcf_model.find_all('light'):
            light.remove()

        # Get wing joint indices into agent's action vector.
        self._wing_inds_action = self._walker._action_indices['wings']
        # Get 'user' index into agent's action vector (only one user action).
        self._user_idx_action = self._walker._action_indices['user'][0]

        # Dummy initialization.
        self._target_height = 0.
        self._target_speed = 0.

        self._target_zaxis = None
        self._ncol = None
        self._grid_axis = None

        # === Explicitly add/enable/disable vision task observables.
        # Fly observables.
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        self._walker.observables.thorax_height.enabled = False
        # Task observables.
        self._walker.observables.add_observable('task_input', self.task_input)
        self._walker.observables.add_observable('Left_LMCs', self.Left_LMCs)
        self._walker.observables.add_observable('Right_LMCs', self.Right_LMCs)

    def get_hfield_height(self, x, y, physics):
        """Return hfield height at a hfield grid point closest to (x, y)."""

        hfield_half_size = physics.model.hfield_size[0, 0]
        self._ncol = physics.model.hfield_ncol[0]
        self._grid_axis = np.linspace(-hfield_half_size, hfield_half_size,
                                      self._ncol)

        # Get nearest indices.
        x_idx = np.argmin(np.abs(self._grid_axis - x))
        y_idx = np.argmin(np.abs(self._grid_axis - y))
        # physics.model.hfield_data is in range [0, 1], needs to be rescaled.
        elevation_z = physics.model.hfield_size[0, 2]  # z_top scaling.
        return elevation_z * physics.model.hfield_data[y_idx * self._ncol +
                                                       x_idx]

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        self._target_height = random_state.uniform(*self._target_height_range)
        self._target_speed = random_state.uniform(*self._target_speed_range)

        theta = np.deg2rad(self._body_pitch_angle)
        self._target_zaxis = np.array([np.sin(theta), 0, np.cos(theta)])

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker.

        Environment call sequence:
            check_termination, get_reward_factors, get_discount
        """
        super().initialize_episode(physics, random_state)

        init_x = random_state.uniform(*self._init_pos_x_range)
        init_y = random_state.uniform(*self._init_pos_y_range)

        # Reset wing pattern generator and get initial wing angles.
        initial_phase = random_state.uniform()
        init_wing_qpos = self._wbpg.reset(initial_phase=initial_phase)

        self._arena.initialize_episode(physics, random_state)

        # Initialize root position and orientation.
        hfield_height = self.get_hfield_height(init_x, init_y, physics)
        init_z = hfield_height + self._target_height
        self._walker.set_pose(physics, np.array([init_x, init_y, init_z]),
                              neg_quat(self._up_dir))

        # Initialize wing qpos.
        physics.bind(self._wing_joints).qpos = init_wing_qpos

        # If enabled, initialize leg joint angles in retracted position.
        if self._leg_joints:
            physics.bind(self._leg_joints).qpos = self._leg_springrefs

        if self._initialize_qvel:
            # Only initialize linear CoM velocity, not rotational velocity.
            init_vel, _ = self._walker.get_velocity(physics)
            self._walker.set_velocity(
                physics, [self._target_speed, init_vel[1], init_vel[2]])

    def before_step(self, physics: 'mjcf.Physics', action, random_state: np.random.RandomState):
        # Get target wing joint angles at beat frequency requested by the agent.
        base_freq, rel_range = self._wbpg.base_beat_freq, self._wbpg.rel_freq_range
        act = action[self._user_idx_action]  # Action in [-1, 1].
        ctrl_freq = base_freq * (1 + rel_range * act)
        ctrl = self._wbpg.step(ctrl_freq=ctrl_freq)  # Returns position control.

        length = physics.bind(self._wing_joints).qpos
        # Convert position control to force control.
        action[self._wing_inds_action] += (ctrl - length)

        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns the factorized reward terms."""

        # Height.
        xpos, _ = self._walker.get_pose(physics)
        current_height = (xpos[2] - self.get_hfield_height(*xpos[:2], physics))
        height = rewards.tolerance(current_height,
                                   bounds=(self._target_height,
                                           self._target_height),
                                   sigmoid='linear',
                                   margin=0.15,
                                   value_at_margin=0)

        velocity, _ = self._walker.get_velocity(physics)

        # Center-of-trench reward factor.
        center_of_trench = 1.
        if isinstance(self._arena, SineTrench):
            trench_specs = self._arena.trench_specs
            # If we are within the trench bounds.
            if trench_specs['x_coords'][0] <= xpos[0] <= trench_specs[
                    'x_coords'][-1]:
                idx = (np.abs(trench_specs['x_coords'] - xpos[0])).argmin()
                trench_center = trench_specs['y_coords'][idx]
                center_of_trench = rewards.tolerance(
                    xpos[1],
                    bounds=(trench_center, trench_center),
                    sigmoid='linear',
                    margin=0.15,
                    value_at_margin=0.0)

        # Preferred absolute flight direction.
        x_speed = rewards.tolerance(velocity[0],
                                    bounds=(self._target_speed, float('inf')),
                                    sigmoid='linear',
                                    margin=1.1 * self._target_speed,
                                    value_at_margin=0.0)

        # Maintain certain speed.
        speed = rewards.tolerance(np.linalg.norm(velocity),
                                  bounds=(self._target_speed,
                                          self._target_speed),
                                  sigmoid='linear',
                                  margin=1.1 * self._target_speed,
                                  value_at_margin=0.0)

        # Keep zero egocentric side speed.
        vel = self.observables['walker/velocimeter'](physics)
        side_speed = rewards.tolerance(vel[1],
                                       bounds=(0, 0),
                                       sigmoid='linear',
                                       margin=10,
                                       value_at_margin=0.0)

        # World z-axis, to replace root quaternion reward above.
        current_zaxis = self.observables['walker/world_zaxis'](physics)
        angle = np.arccos(np.dot(self._target_zaxis, current_zaxis))
        world_zaxis = rewards.tolerance(angle,
                                        bounds=(0, 0),
                                        sigmoid='linear',
                                        margin=np.pi,
                                        value_at_margin=0.0)

        # Reward for leg retraction during flight.
        qpos_diff = physics.bind(self._leg_joints).qpos - self._leg_springrefs
        retract_legs = rewards.tolerance(qpos_diff,
                                         bounds=(0, 0),
                                         sigmoid='linear',
                                         margin=4.,
                                         value_at_margin=0.0)

        return np.hstack((height, x_speed, speed, side_speed, world_zaxis,
                          center_of_trench, retract_legs))


    def check_floor_contact(self, physics):
        """Check if fly collides with floor geom."""
        world_id = 0
        for contact in physics.data.contact:
            # If the contact is not active, continue.
            if contact.efc_address < 0:
                continue
            # Check floor contact.
            body1 = physics.model.geom_bodyid[contact.geom1]
            body2 = physics.model.geom_bodyid[contact.geom2]
            if body1 == world_id or body2 == world_id:
                return True
        return False

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        if self._floor_contacts_fatal:
            return (self.check_floor_contact(physics)
                    or super().check_termination(physics))
        else:
            return super().check_termination(physics)

    @property
    def target_height(self):
        return self._target_height

    @property
    def target_speed(self):
        return self._target_speed

    @composer.observable
    def task_input(self):
        """Task-specific input, framed as an observable."""
        def get_task_input(physics: 'mjcf.Physics'):
            del physics
            return np.hstack([self._target_height, self._target_speed])
        
        return observable.Generic(get_task_input)

    @composer.observable
    def Left_LMCs(self):
        def get_left_lmcs(physics: 'mjcf.Physics'):
            left_eye = self.observables['walker/left_eye'](physics)
            left_eye = tf.cast(left_eye, tf.float32) / 255.0  # <- cast + 归一化
            if left_eye.shape[-1] == 3:
                left_eye = tf.reduce_mean(left_eye, axis=-1)  # 灰度化
            return self._lmc_left(left_eye)

        return observable.Generic(get_left_lmcs)

    @composer.observable
    def Right_LMCs(self):
        def get_right_lmcs(physics: 'mjcf.Physics'):
            right_eye = self.observables['walker/right_eye'](physics)

            # cast + 归一化
            right_eye = tf.cast(right_eye, tf.float32) / 255.0

            # ⚠️ 在灰度化前做左右翻转
            right_eye = tf.image.flip_left_right(right_eye)

            # 灰度化
            if right_eye.shape[-1] == 3:
                right_eye = tf.reduce_mean(right_eye, axis=-1)

            return self._lmc_right(right_eye)

        return observable.Generic(get_right_lmcs)


class LMCsProcessor(snt.Module):
    """L1-L3 neuron response preprocessor (single image, no batch)."""
    def __init__(self, time_step=1, tau_l1=9, tau_l2=9, tau_l3=6,
                 visualize=True, visualize_every=1, save_dir="lmc_vis"):
        super().__init__()
        self.time_step = time_step
        self.tau_l1 = tau_l1
        self.tau_l2 = tau_l2
        self.tau_l3 = tau_l3
        self.visualize = visualize
        self.visualize_every = visualize_every
        self.save_dir = save_dir
        self.frame_count = 0

        self.l1_state = None
        self.l2_state = None
        self.last_frame = None

        self.params = {
            'L1_B': {'a': -0.917, 'b': 1.992},
            'L1_D': {'a': -2.326, 'b': -5.377},
            'L2_B': {'a': -0.814, 'b': 1.950},
            'L2_D': {'a': -2.044, 'b': -3.623},
        }

    def _sigmoid(self, x, a, b):
        return a * x / (1.0 + tf.abs(b * x))

    def __call__(self, current_frame):
        current = tf.cast(current_frame, tf.float32)
        self.frame_count += 1

        if self.l1_state is None:
            shape = current.shape
            self.l1_state = tf.Variable(tf.zeros(shape), trainable=False)
            self.l2_state = tf.Variable(tf.zeros(shape), trainable=False)
            self.last_frame = tf.Variable(tf.zeros(shape), trainable=False)

        # --- contrast ---
        contrast = tf.where(
            self.last_frame != 0,
            (current - self.last_frame) / (self.last_frame + 1e-6),
            tf.zeros_like(current)
        )

        # 对上一帧为0且当前帧>0的位置，赋值1e4
        contrast = tf.where(
            (self.last_frame == 0) & (current > 0),
            tf.constant(1e4, dtype=tf.float32),
            contrast
        )

        direction = tf.where(current > self.last_frame, 1, -1)

        # --- L1 ---
        mask_B = direction == 1
        mask_D = direction == -1
        delta_l1_B = self._sigmoid(contrast, self.params['L1_B']['a'], self.params['L1_B']['b'])
        delta_l1_D = self._sigmoid(contrast, self.params['L1_D']['a'], self.params['L1_D']['b'])
        delta_l1 = tf.where(mask_B, delta_l1_B, tf.zeros_like(delta_l1_B))
        delta_l1 = tf.where(mask_D, delta_l1_D, delta_l1)

        static_l1 = 0.35 * tf.exp(-2.36 * current) + 0.08
        self.l1_state.assign(self.l1_state * tf.exp(-self.time_step / self.tau_l1) + delta_l1)
        l1_out = self.l1_state + static_l1

        # --- L2 ---
        delta_l2_B = self._sigmoid(contrast, self.params['L2_B']['a'], self.params['L2_B']['b'])
        delta_l2_D = self._sigmoid(contrast, self.params['L2_D']['a'], self.params['L2_D']['b'])
        delta_l2 = tf.where(mask_B, delta_l2_B, tf.zeros_like(delta_l2_B))
        delta_l2 = tf.where(mask_D, delta_l2_D, delta_l2)

        self.l2_state.assign(self.l2_state * tf.exp(-self.time_step / self.tau_l2) + delta_l2)
        l2_out = self.l2_state

        # --- L3 ---
        l3_out = 0.62 * tf.exp(-2.9 * current) + 0.08
        prev_frame = tf.identity(self.last_frame)

        # --- 更新上一帧 ---
        output = tf.stack([
            tf.cast(l1_out, tf.float32),
            tf.cast(l2_out, tf.float32),
            tf.cast(l3_out, tf.float32)
        ], axis=-1)

        self.last_frame.assign(current)

        # --- 可视化 ---
        # if self.visualize and self.frame_count % self.visualize_every == 0:
        #     self._visualize_full(prev_frame, current, contrast, output, self.frame_count)

        return output

    def _visualize_full(self, prev_frame, current, contrast, output, step_idx):
        """输出：上一帧、当前帧、contrast、L1、L2、L3"""
        os.makedirs(self.save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        imgs = [
            (prev_frame.numpy(), "Prev Gray"),
            (current.numpy(), "Current Gray"),
            (contrast.numpy(), "Contrast"),
            (output[..., 0].numpy(), "L1"),
            (output[..., 1].numpy(), "L2"),
            (output[..., 2].numpy(), "L3"),
        ]

        for ax, (img, title) in zip(axes.ravel(), imgs):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        file_path = os.path.join(self.save_dir, f"lmc_full_{step_idx:05d}.png")
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"💾 Saved full LMC visualization → {file_path}")
