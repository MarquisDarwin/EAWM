import datetime
import gym
import numpy as np
import uuid
import envs.dmcgb as dmcgb
import torch
from dm_env import StepType, specs
import dm_control
import dm_env
import os
import cv2
import xmltodict
import copy
import collections
from PIL import Image
from dm_control.mujoco.wrapper import mjbindings
class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()

dmcgb_photometric_modes = ['color_easy', 'color_hard', 'video_easy', 'video_hard', 'color_video_easy', 'color_video_hard']
dmcgb_data_dir = "envs/data"


class ColorVideoWrapper(dm_env.Environment):
    ''' DMCGB Wrapper for dmcontrol suite for applying changes in colors and videos, must be applied before pixel wrapper'''
    def __init__(self, env, mode, seed, video_render_size=256):
        self._env = env
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._video_render_size = video_render_size

        # XML of current domain
        self._xml = self._get_model_and_assets(self._env._domain_name+'.xml')
        
        # Video
        self._video_paths = []
        self._current_video_frame = 0 # Which frame in video, placeholder
        self._current_video_len = 1 # Length of video, placeholder
        self._SKY_TEXTURE_INDEX = 2 # Default skybox
        self._Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))

        # Mode
        self._color_in_effect= 'color' in self._mode
        self._video_in_effect= 'video' in self._mode
        self._remove_ground_and_rails = (self._mode == 'video_hard') 
        self._moving_domain = self._env._domain_name in ['walker', 'cheetah'] # Background needs to move with them
        self._moving_domain_offset_x = 0 if self._env._domain_name == 'walker' else -0.05 # Walker or Cheetah 
        self._moving_domain_offset_z = -1.07 if self._env._domain_name == 'walker' else 0.15

        # Loading 
        start_index = self._random_state.randint(100)
        if self._color_in_effect:
            self._load_colors() # Get Colors
            self._num_colors = len(self._colors)
            assert self._num_colors >= 100, 'env must include at least 100 colors'
            self._color_index =  start_index % self._num_colors
        if self._video_in_effect:
            self._get_video_paths() # Get Videos
            self._num_videos = len(self._video_paths)
            self._video_index = start_index % self._num_videos
            self._reload_physics(*self._reformat_xml({})) # Create backcube with video




# -------------------Video Helpers--------------------------------------------

    def _get_video_paths(self):
        if 'easy' in self._mode:
            video_dir = os.path.join(dmcgb_data_dir, 'video_easy')
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
        elif 'hard' in self._mode:
            video_dir = os.path.join(dmcgb_data_dir, 'video_hard')
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
        i, ret = 0, True
        while (i < n  and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return buf


    # This function forces the wrapper to be before the environment is rendered in the pixel wrapper
    def _move_backcube(self): 
        """ Moves the backcube to follow camera frame when dealing with moving domains like cheetah and walker """
        if self._moving_domain:
            body_x_pos = self._env.physics.data.body('torso').subtree_com[0]
            body_z_pos = self._env.physics.data.body('torso').subtree_com[2]
            self._env.physics.data.site('video_screen').xpos[0] = body_x_pos + self._moving_domain_offset_x
            self._env.physics.data.site('video_screen').xpos[2] = body_z_pos + self._moving_domain_offset_z


    def _reset_background(self, render_size=None):
        """ Sets the stage for video background in simulation and loads and prepares the video images """
        # Extra things to remove
        if self._remove_ground_and_rails:
            self._env.physics.named.model.mat_rgba['grid', 'a'] = 0 # Removing grid

        # Set image size in simulation
        if render_size is not None:
            self._video_render_size = render_size
        sky_height = self._env.physics.model.tex_height[self._SKY_TEXTURE_INDEX] = sky_width = self._env.physics.model.tex_width[self._SKY_TEXTURE_INDEX]=self._video_render_size
        sky_size = sky_height * sky_width * 3
        sky_address = self._env.physics.model.tex_adr[self._SKY_TEXTURE_INDEX]
        # Load images from video
        self._video_index = (self._video_index + 1) % self._num_videos
        images = self._load_video(self._video_paths[self._video_index])
        self._current_video_len = len(images)
        # Generate image textures
        texturized_images = []
        for image in images:
            image_flattened = self._size_and_flatten(image, sky_height, sky_width)
            texturized_images.append(image_flattened)
        self._background = self._Texture(sky_size, sky_address, texturized_images)


    def _apply_video(self):
        """Apply the background video texture to the backcube and increment counter"""
        assert self._background is not None, "Missing reference to skybox background in VideoWrapper"
        start = self._background.address
        end = self._background.address + self._background.size
        texture = self._background.textures[self._current_video_frame]
        self._env.physics.model.tex_rgb[start:end] = texture
        # Upload the new texture to the GPU. 
        with self._env.physics.contexts.gl.make_current() as ctx:
            ctx.call(
                mjbindings.mjlib.mjr_uploadTexture,
                self._env.physics.model.ptr,
                self._env.physics.contexts.mujoco.ptr,
                self._SKY_TEXTURE_INDEX,
            )
        # Increment
        self._current_video_frame = (self._current_video_frame + 1) % self._current_video_len
    

    def _size_and_flatten(self, image, ref_height, ref_width):
        """ Resize image if necessary and flatten the result """
        image_height, image_width = image.shape[:2]
        if image_height != ref_height or image_width != ref_width:
            image = np.asarray(Image.fromarray(image).resize(size=(ref_width, ref_height)))
        return image.flatten(order='K')


    def _render_high(self, size=256, camera_id=0):
        """ 
        Utility function to override original set background video resolution with a higher resolution for recording
        This function changes the background videos resolution for the environment, which will slow the speed of the env.
        """
        if size != self._video_render_size:
            self._reset_background(render_size=size)
            self._video_render_size = size
        return self._env.physics.render(height=size, width=size, camera_id=camera_id)

# -------------------------------Color helpers------------------------------------

    def _load_colors(self):
        if 'hard' in self._mode: 
            self._colors = torch.load(f'{dmcgb_data_dir}/color_hard.pt',weights_only=False)
        elif 'easy' in self._mode:
            self._colors = torch.load(f'{dmcgb_data_dir}/color_easy.pt',weights_only=False)

    def _randomize_colors(self):
        chosen_colors =  self._colors[self._color_index]
        self._reload_physics(*self._reformat_xml(chosen_colors))
        self._color_index = (self._color_index + 1) % self._num_colors
        

    def _reload_physics(self, xml_string, assets=None):
        assert hasattr(self._env, 'physics'), 'environment does not have physics attribute'
        # For newer mujoco need to convert from str to bytes
        if assets:
            new_assets = {}
            for key, val in assets.items():
                if type(val) == bytes:
                    new_assets[key] = val
                else:
                    new_assets[key] = val.encode('utf-8')
            assets = new_assets
        self._env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_model_and_assets(self, model_fname):
        """"Returns a tuple containing the model XML string and a dict of assets."""
        # ball_in_cup different name
        if model_fname == "cup.xml":
            model_fname = "ball_in_cup.xml"
        # Convert XML to dicts
        model = dm_control.suite.common.read_model(model_fname)
        assets = dm_control.suite.common.ASSETS
        return model, assets


    def _reformat_xml(self, chosen_colors):
        model_xml, assets = self._xml
        model_xml = copy.deepcopy(model_xml)
        assets = copy.deepcopy(assets)

        # Convert XML to dicts
        model = xmltodict.parse(model_xml)
        materials = xmltodict.parse(assets['./common/materials.xml'])
        skybox = xmltodict.parse(assets['./common/skybox.xml'])

        # Edit grid floor
        if 'grid_rgb1' in chosen_colors:
            assert isinstance(chosen_colors['grid_rgb1'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@rgb1'] = \
                f'{chosen_colors["grid_rgb1"][0]} {chosen_colors["grid_rgb1"][1]} {chosen_colors["grid_rgb1"][2]}'
        if 'grid_rgb2' in chosen_colors:
            assert isinstance(chosen_colors['grid_rgb2'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@rgb2'] = \
                f'{chosen_colors["grid_rgb2"][0]} {chosen_colors["grid_rgb2"][1]} {chosen_colors["grid_rgb2"][2]}'
        if 'grid_markrgb' in chosen_colors:
            assert isinstance(chosen_colors['grid_markrgb'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@markrgb'] = \
                f'{chosen_colors["grid_markrgb"][0]} {chosen_colors["grid_markrgb"][1]} {chosen_colors["grid_markrgb"][2]}'
        if 'grid_texrepeat' in chosen_colors:
            assert isinstance(chosen_colors['grid_texrepeat'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
                f'{chosen_colors["grid_texrepeat"][0]} {chosen_colors["grid_texrepeat"][1]}'

        # Edit self
        if 'self_rgb' in chosen_colors:
            assert isinstance(chosen_colors['self_rgb'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
            materials['mujoco']['asset']['material'][1]['@rgba'] = \
                f'{chosen_colors["self_rgb"][0]} {chosen_colors["self_rgb"][1]} {chosen_colors["self_rgb"][2]} 1'

        # Edit skybox
        if 'skybox_rgb' in chosen_colors:
            assert isinstance(chosen_colors['skybox_rgb'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@rgb1'] = \
                f'{chosen_colors["skybox_rgb"][0]} {chosen_colors["skybox_rgb"][1]} {chosen_colors["skybox_rgb"][2]}'
        if 'skybox_rgb2' in chosen_colors:
            assert isinstance(chosen_colors['skybox_rgb2'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@rgb2'] = \
                f'{chosen_colors["skybox_rgb2"][0]} {chosen_colors["skybox_rgb2"][1]} {chosen_colors["skybox_rgb2"][2]}'
        if 'skybox_markrgb' in chosen_colors:
            assert isinstance(chosen_colors['skybox_markrgb'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@markrgb'] = \
                f'{chosen_colors["skybox_markrgb"][0]} {chosen_colors["skybox_markrgb"][1]} {chosen_colors["skybox_markrgb"][2]}'

        # For Videos Add a Cube/Box Behind Model to Project Videos on
        if self._video_in_effect:
            domain = self._env._domain_name

            # Adding texture to update with videos
            materials['mujoco']['asset']['texture'] = [ materials['mujoco']['asset']['texture'],
                                                       {'@name':'projector', '@type':'skybox', '@builtin':'flat', '@width':'512', '@height':'512', '@mark':'none'}]
            materials['mujoco']['asset']['material'].append({'@name':'projector', '@texture':'projector', '@texuniform':'false', '@specular':'0', '@shininess':'0', '@reflectance':'0', '@emission':'0'})

            # Projector Cubes to add as sites
            site_dicts = {
                'walker': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 1.96 -0.7', '@euler': '-30 0 0', '@material':'projector'},
                'cheetah': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 1.6 -0.7', '@euler': '0 0 0', '@material':'projector'},
                'cartpole': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 0.5 1.0', '@euler': '0 0 0', '@material':'projector'},
                'ball_in_cup': {'@name':'video_screen', '@type':'box', '@size':'0.66 0.1 0.66', '@pos':'0.0 0.5 0.05', '@euler': '-27 0 0', '@material':'projector'},
                'finger': {'@name':'video_screen', '@type':'box', '@size':'0.66 0.1 0.66', '@pos':'0.0 0.5 0.05', '@euler': '-27 0 0', '@material':'projector'},
            }

            # Adding sites to xmls
            if domain in ['walker', 'cheetah', 'cartpole', 'ball_in_cup']:
                model['mujoco']['worldbody']['site'] = site_dicts[domain]
            elif domain in ['finger']:
                model['mujoco']['worldbody']['site'] = [model['mujoco']['worldbody']['site'], site_dicts[domain]]


        # Convert back to XML
        model_xml = xmltodict.unparse(model)
        assets['./common/materials.xml'] = xmltodict.unparse(materials)
        assets['./common/skybox.xml'] = xmltodict.unparse(skybox)

        return model_xml, assets


# --------------------------------Main functions--------------------------------
    def reset(self):
        """Reset the background state."""
        if self._color_in_effect: # loads a color and resets env with xml
            self._randomize_colors()
        time_step = self._env.reset()
        if self._video_in_effect: # removes backgrounds and updates textures/backcube
            self._current_video_frame = 0
            self._reset_background()
            self._apply_video()
            self._move_backcube()
        return time_step
    
    def step(self, action):
        time_step = self._env.step(action)
        if self._video_in_effect:
            self._apply_video()
            self._move_backcube()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
dmcgb_geometric_modes= ['rotate_easy', 'rotate_hard', 'shift_easy', 'shift_hard', 'rotate_shift_easy', 'rotate_shift_hard'] 
CAMERA_MODES = ['fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom']
VALID_MODES = dmcgb_photometric_modes + dmcgb_geometric_modes

class ShiftWrapper(dm_env.Environment):
    """Shifts camera by rotating its lookat point to push the agent towards the edges of the image"""
    def __init__(self, env, mode, seed):
        self._env = env
        self._random_state_shift = np.random.RandomState(seed)
        self._curr_cam_shift = np.zeros((3,3))
        self._start_shift_ind = 0
        self._num_starting_shifts = 100
        self._curr_cam_mode = CAMERA_MODES[self._env.physics.model.cam_mode[0]]
        self._shift_in_effect = "shift" in mode

        if self._shift_in_effect:
            self._get_corner_coordinates(mode)
            self._sample_cam_positions()
            self._random_state_shift.shuffle(self._all_cam_shifts) 



    def _get_corner_coordinates(self, mode):
        cam_edges = { 
            'walker': [10, -10, 10, -10], # Right, Left, Down, Up
            'cheetah': [10, -10, 10, -10], # Right, Left, Down, Up
            'cartpole': [8, -8, 10, -10], # Left, Right, Up, Down
            'finger': [6, -6, 12, -6], # Left, Right, Up, Down
            'ball_in_cup': [6, -6, 12, -6], # Left, Right, Up, Down
        }
        max_roll, min_roll, max_pitch, min_pitch = cam_edges[self._env._domain_name] 
        if "easy" in mode:
            max_roll /= 1.5
            min_roll /= 1.5
            max_pitch /= 1.5
            min_pitch /= 1.5
        self._four_corners = np.array([[0, min_roll, min_pitch], [0, max_roll, min_pitch],[0, max_roll, max_pitch],  [0, min_roll, max_pitch]]) 


    def _sample_cam_positions(self):
        # Interpolating rotations between four corners
        num_points_each_path = self._num_starting_shifts // 4
        positions = []
        for i in range(4):
            next_i = (i + 1) % 4 
            diff = self._four_corners[next_i] - self._four_corners[i]
            scale = diff / num_points_each_path
            sampled_positions = (np.arange(num_points_each_path)[..., np.newaxis] * scale) + self._four_corners[i]
            positions.append(sampled_positions)
        positions = np.concatenate(positions, axis=0)
        
        # Converting to rotation mats
        shifts =[]
        for rot_combo in positions:
            yaw, pitch, roll = rot_combo
            # Convert angles to radians
            yaw = np.radians(yaw)
            pitch = np.radians(pitch)
            roll = np.radians(roll)

            # Individual rotation matrices
            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

            R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                [0, 1, 0],
                                [-np.sin(pitch), 0, np.cos(pitch)]])

            R_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

            # Combine rotation matrices
            rotation_matrix = np.dot(R_yaw, np.dot(R_pitch, R_roll))
            shifts.append(rotation_matrix)
        # Appending extra to fill required starting positions
        for i in range(self._num_starting_shifts - len(shifts)):
            shifts.append(shifts[i])
        # Store them
        self._all_cam_shifts = shifts


    def _set_cam_shift(self):
        # Set Camera Shift
        cam_xmat = np.reshape(self._env.physics.data.cam_xmat[0], (3,3))
        self._env.physics.data.cam_xmat[0] = np.dot(cam_xmat, self._curr_cam_shift).flatten() # Slide to view

    def reset(self):
        time_step = self._env.reset()
        if self._shift_in_effect:
            self._curr_cam_shift[:] = self._all_cam_shifts[self._start_shift_ind][:]
            self._start_shift_ind = (self._start_shift_ind + 1) % len(self._all_cam_shifts) # Next time new cam position
            self._set_cam_shift()
        return time_step 

    def step(self, action):
        time_step = self._env.step(action)
        if self._shift_in_effect:
            self._set_cam_shift()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RotateWrapper(dm_env.Environment):
    """Rotates the frame by rotating the camera's yaw 360 deg"""
    def __init__(self, env, mode, seed):
        self._env = env
        self._rotate_in_effect = "rotate" in mode
        if self._rotate_in_effect:
            self._random_state_rot = np.random.RandomState(seed)
            self._start_rot_ind = 0
            self._num_starting_rots = 100 
            self._curr_cam_rot = np.zeros((3,3))
            self._get_rotation_angles(mode)
            self._random_state_rot.shuffle(self._all_cam_rots) 


    def _get_rotation_angles(self, mode):
        bound_angles = np.array([-180.0, 180.0])
        if "easy" in mode:
            bound_angles /= 2.0 
        min_angle, max_angle = bound_angles
        scale = (max_angle - min_angle) / self._num_starting_rots
        rot_angles = (np.arange(self._num_starting_rots) * scale) + min_angle
        # Converting to rotation mats
        rots =[]
        for yaw in rot_angles:
            yaw = np.radians(yaw)
            # Individual rotation matrices
            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
            rots.append(R_yaw)
        # Appending extra to fill required starting positions
        for i in range(self._num_starting_rots - len(rots)):
            rots.append(rots[i])
        self._all_cam_rots = rots 


    def _set_cam_rot(self):
        # Set Camera Rotation
        cam_xmat = np.reshape(self._env.physics.data.cam_xmat[0], (3,3))
        self._env.physics.data.cam_xmat[0] = np.dot(cam_xmat, self._curr_cam_rot).flatten() # Slide to view


    def reset(self):
        time_step = self._env.reset()
        if self._rotate_in_effect:
            self._curr_cam_rot = self._all_cam_rots[self._start_rot_ind]
            self._start_rot_ind = (self._start_rot_ind + 1) % len(self._all_cam_rots) # Next time new cam position
            self._set_cam_rot()
        return time_step 


    def step(self, action):
        time_step = self._env.step(action)
        if self._rotate_in_effect:
            self._set_cam_rot()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)