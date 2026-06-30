
"""
AGIBOT G1 真实观测推理测试
核心逻辑:
1. 推理 → 执行动作序列 → 等到到达目标 → 立即推理
2. 超时只是警告，不会停止执行
3. 到达后立即推理，不额外等待
"""
import numpy as np
import cv2
import msgpack
import msgpack_numpy as mnp
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState #ROS2官方库
from cv_bridge import CvBridge
import time
import os
from datetime import datetime
from scipy.interpolate import make_interp_spline # SciPy 的「插值子模块」，专门处理离散数据的平滑补点
try:
    from genie_msgs.msg import EndState #智元机器人官方库
except ImportError:
    EndState = None


REQUIRED_ACTION_KEYS = ("joint_position", "left_effector_position", "right_effector_position")
EXPECTED_JOINT_DIM = 14
EXPECTED_GRIPPER_DIM = 1


def format_vector(values, precision=4):
    """Format a numpy/list vector for logging; numpy arrays do not support {:.4f}."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.array2string(
        arr,
        separator=", ",
        formatter={"float_kind": lambda x: f"{x:+.{precision}f}"},
    )


def key_tail(key):
    return str(key).split(".")[-1].lower()


def canonical_action_key(key):
    lower_key = str(key).lower()
    tail = key_tail(lower_key)
    if tail == "joint_position" or lower_key.endswith(".joint_position"):
        return "joint_position"
    if tail == "left_effector_position" or lower_key.endswith(".left_effector_position"):
        return "left_effector_position"
    if tail == "right_effector_position" or lower_key.endswith(".right_effector_position"):
        return "right_effector_position"
    return str(key)


def adapt_action_for_control(action):
    """Adapt server action keys to the format expected by the existing control code."""
    normalized = {}
    for key, value in action.items():
        normalized_key = key[len("action.") :] if key.startswith("action.") else key
        normalized[canonical_action_key(normalized_key)] = np.asarray(value, dtype=np.float32)

    # N1.7 checkpoint returns arm actions split by side. Keep the existing control logic
    # unchanged by adapting them back to the 14-DoF joint_position layout used below.
    if (
        "joint_position" not in normalized
        and "left_arm_joint_position" in normalized
        and "right_arm_joint_position" in normalized
    ):
        left_arm = normalized["left_arm_joint_position"]
        right_arm = normalized["right_arm_joint_position"]
        if left_arm.ndim != 3 or right_arm.ndim != 3:
            raise ValueError(
                "left_arm_joint_position/right_arm_joint_position 需要是 "
                f"(batch, horizon, dim) 三维数组，实际分别为 {left_arm.shape}, {right_arm.shape}"
            )
        if left_arm.shape[:2] != right_arm.shape[:2]:
            raise ValueError(
                "left_arm_joint_position/right_arm_joint_position 的 batch/horizon 不一致，"
                f"实际分别为 {left_arm.shape}, {right_arm.shape}"
            )
        normalized["joint_position"] = np.concatenate([left_arm, right_arm], axis=-1)

    missing = [key for key in REQUIRED_ACTION_KEYS if key not in normalized]
    if missing:
        raise KeyError(
            "推理服务返回的 action 缺少脚本需要的键: "
            f"{missing}; 当前 action keys: {list(action.keys())}. "
            "脚本支持直接返回 joint_position，或返回 "
            "left_arm_joint_position/right_arm_joint_position 后自动拼接。"
        )

    for key in REQUIRED_ACTION_KEYS:
        if normalized[key].ndim != 3:
            raise ValueError(
                f"action['{key}'] 需要是 (batch, horizon, dim) 三维数组，"
                f"实际 shape={normalized[key].shape}"
            )

    if normalized["joint_position"].shape[-1] != EXPECTED_JOINT_DIM:
        raise ValueError(
            "当前脚本只会发布 14 个手臂关节，但服务端返回的 "
            f"action['joint_position'] shape={normalized['joint_position'].shape}。"
        )

    for key in ("left_effector_position", "right_effector_position"):
        if normalized[key].shape[-1] != EXPECTED_GRIPPER_DIM:
            raise ValueError(
                f"当前脚本期望 action['{key}'] 最后一维为 1，"
                f"实际 shape={normalized[key].shape}。"
            )

    return normalized


def print_action_summary(name, arr, current=None):
    arr = np.asarray(arr, dtype=np.float32)
    print(f"   {name}: shape={arr.shape}, dtype={arr.dtype}, min={np.min(arr):+.4f}, max={np.max(arr):+.4f}, mean={np.mean(arr):+.4f}")
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    traj = arr[0]
    print(f"      first: {format_vector(traj[0])}")
    print(f"      last : {format_vector(traj[-1])}")
    print(f"      last-first: {format_vector(traj[-1] - traj[0])}")
    if traj.shape[0] > 1:
        step_delta = np.diff(traj, axis=0)
        max_step_delta = np.max(np.abs(step_delta), axis=0)
        print(f"      max |step delta|: {format_vector(max_step_delta)}")
    if current is not None:
        current = np.asarray(current, dtype=np.float32)
        if current.shape[-1] == traj.shape[-1]:
            print(f"      last-current: {format_vector(traj[-1] - current)}")


def print_model_output(raw_action, adapted_action, current_joints):
    print("\n📦 模型原始输出:")
    print(f"   raw action keys: {list(raw_action.keys())}")
    current_joints = np.asarray(current_joints, dtype=np.float32)
    current_by_key = {
        "left_arm_joint_position": current_joints[:7],
        "right_arm_joint_position": current_joints[7:14],
        "joint_position": current_joints[:14],
    }
    for key, value in raw_action.items():
        print_action_summary(key, value, current=current_by_key.get(key))

    print("\n🔎 适配给原控制逻辑后的动作:")
    print_action_summary("joint_position", adapted_action["joint_position"], current=current_joints[:14])
    print_action_summary("left_effector_position", adapted_action["left_effector_position"])
    print_action_summary("right_effector_position", adapted_action["right_effector_position"])
    print("   当前为只观察模式：不会发布任何 /wbc 控制指令。")


class PolicyClient:
    """GR00T N1.7-compatible PolicyClient using msgpack_numpy."""

    def __init__(self, host="localhost", port=5555, timeout_ms=15000, api_token=None):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()
        print("✅ 使用 GR00T N1.7 协议 PolicyClient (msgpack_numpy)")

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _to_bytes(data):
        return msgpack.packb(data, default=mnp.encode, use_bin_type=True)

    @staticmethod
    def _from_bytes(data):
        return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(self._to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the correct policy server is running.")

        response = self._from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observation, options=None):
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        action, info = tuple(response)
        return {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}, info

    def get_modality_config(self):
        return self.call_endpoint("get_modality_config", requires_input=False)

    def ping(self):
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False


class RobotObservationNode(Node):
    """机器人观测采集节点"""

    def __init__(self, enable_control=False):
        super().__init__('robot_observation_node') #初始化节点名称为 'robot_observation_node'
        self.bridge = CvBridge() # 初始化 CvBridge 对象，用于将 ROS 的 Image 消息转换为 OpenCV 图像格式
        self.enable_control = enable_control # 保存传入的控制开关标志。如果为 True，节点将发布控制指令；如果为 False，则仅采集数据。
    
        # 存储最新数据
        self.latest_images = {} # 用于存储来自相机话题的最新图像数据（RGB 格式）
        self.latest_joint_state = None # 用于存储最新的 14 个机械臂关节位置（左右臂各 7 个）
        self.latest_left_gripper = None # 用于存储左夹爪的最新位置（单位通常为 mm）
        self.latest_right_gripper = None # 用于存储右夹爪的最新位置（单位通常为 mm）

        # 存储动作序列中最后一个动作
        self.last_joint_pos = None     # 用于存储上一次动作序列执行后的最终关节位置。目的：在下一次推理开始前，计算从"上次结束位置"到"本次开始位置"的过渡轨迹，避免动作突变。
        self.last_left_gripper = None  # scalar  用于存储上一次动作序列执行后的最终左夹爪状态
        self.last_right_gripper = None # scalar 用于存储上一次动作序列执行后的最终右夹爪状态
       
        # 数据接收标志
        #字典：标记各路传感器数据是否已成功接收至少一次。
        self.data_ready = {
            'head_image': False,
            'left_hand_image': False,
            'right_hand_image': False,
            'joint_state': False,
            'left_gripper': False,
            'right_gripper': False,
        }
       
        print("📡 创建ROS2订阅...")
       
        # 创建订阅者-用于接收话题
        #订阅相机话题，队列大小为 10，回调函数为 _head_callback
        self.create_subscription(Image, '/camera/head_color', self._head_callback, 10)
        self.create_subscription(Image, '/camera/hand_left_color', self._left_hand_callback, 10)
        self.create_subscription(Image, '/camera/hand_right_color', self._right_hand_callback, 10)
        
        # 订阅关节状态话题
        self.create_subscription(JointState, '/hal/arm_joint_state', self._joint_callback, 10)
        
        # 订阅夹爪状态话题
        if EndState is not None: #检查EndState是否已成功导入（智元机器人特有的消息）
            self.create_subscription(EndState, '/hal/left_ee_data', self._left_gripper_callback, 10)
            self.create_subscription(EndState, '/hal/right_ee_data', self._right_gripper_callback, 10)
            print("✅ 夹爪订阅已创建 (必需)")
       
        # 创建发布者-用于发布控制指令
        if self.enable_control: #数据采集完成再启用控制功能，发布控制指令
            self.arm_pub = self.create_publisher(JointState, '/wbc/arm_command', 10)
            self.left_gripper_pub = self.create_publisher(JointState, '/wbc/left_ee_command', 10)
            self.right_gripper_pub = self.create_publisher(JointState, '/wbc/right_ee_command', 10)
            print("✅ 控制发布者已创建")
       
        print("✅ ROS2订阅已创建")
        print("⏳ 等待ROS2连接建立 (2秒)...")
        time.sleep(2.0)
   
   #回调函数：处理头部相机图像话题
    def _head_callback(self, msg):
        try:
            #将 ROS Image 消息解码为 RGB8 格式 (红绿蓝顺序)
            self.latest_images['top_head'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['head_image'] = True #标记头部图像数据已就绪
        except Exception: #如果 RGB8 解码失败（通常是因为源数据是 BGR），进行降级处理
            try:#先将image转换为BGR格式，再转换为RGB格式
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                #将 BGR 图像转换为 RGB 图像，保持数据格式一致性
                self.latest_images['top_head'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['head_image'] = True
            except Exception:
                pass
    #回调函数：处理左手相机图像话题
    def _left_hand_callback(self, msg):
        try:
            self.latest_images['hand_left'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['left_hand_image'] = True
        except Exception:
            try:
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images['hand_left'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['left_hand_image'] = True
            except Exception:
                pass
    #回调函数：处理右手相机图像话题
    def _right_hand_callback(self, msg):
        try:
            self.latest_images['hand_right'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.data_ready['right_hand_image'] = True
        except Exception:
            try:
                bgr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images['hand_right'] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                self.data_ready['right_hand_image'] = True
            except Exception:
                pass
    #回调函数：处理关节状态话题
    def _joint_callback(self, msg):
        try:
            if len(msg.position) >= 14: 
                self.latest_joint_state = msg.position[:14] #只取前 14 个数据，存储最新的关节位置
                self.data_ready['joint_state'] = True
        except Exception as e:
            print(f"关节数据解析错误:{e}")
    #回调函数：处理左夹爪状态话题
    def _left_gripper_callback(self, msg):
        try:
            if len(msg.end_state) > 0: #检查end_state列表是否有数据
                #提取第一个末端执行器的位置信息，并转换为浮点数存储
                self.latest_left_gripper = float(msg.end_state[0].position)
                self.data_ready['left_gripper'] = True
        except Exception as e:
            print(f"⚠️ 左夹爪数据解析错误: {e}")
    #回调函数：处理右夹爪状态话题
    def _right_gripper_callback(self, msg):
        try:
            if len(msg.end_state) > 0:
                #提取第一个末端执行器的位置信息，并转换为浮点数存储
                self.latest_right_gripper = float(msg.end_state[0].position)
                self.data_ready['right_gripper'] = True
        except Exception as e:
            print(f"⚠️ 右夹爪数据解析错误:  {e}")
   
    def is_data_ready(self):
        #检查所有必需的数据源是否都已接收到
        #bool: 如果所有 data_ready 字典中的值都为 True，则返回 True，否则返回 False。
        return all(self.data_ready.values())
    
    #数据接收状态：显示当前各个传感器的数据接收状态。用于在控制台打印日志，方便用户确认系统连接情况。
    def get_data_status(self):
        status_items = []
        #检查并添加各路图像和关节的状态到状态列表
        status_items.append(f"头部图像: {'✓' if self.data_ready['head_image'] else '✗'}")
        status_items.append(f"左手图像: {'✓' if self.data_ready['left_hand_image'] else '✗'}")
        status_items.append(f"右手图像: {'✓' if self.data_ready['right_hand_image'] else '✗'}")
        status_items.append(f"关节状态: {'✓' if self.data_ready['joint_state'] else '✗'}")
        #对于夹爪，如果数据已就绪，额外显示其当前的具体数值
        left_grip_status = f"✓({self.latest_left_gripper:.1f}mm)" if self.data_ready['left_gripper'] else "✗"
        right_grip_status = f"✓({self.latest_right_gripper:.1f}mm)" if self.data_ready['right_gripper'] else "✗"
        status_items.append(f"左夹爪: {left_grip_status}")
        status_items.append(f"右夹爪: {right_grip_status}")
       
        return " | ".join(status_items)
    
    #获取当前缺失的传感器数据列表
    def get_missing_data(self):
        missing = []
        # 定义内部键到中文名称的映射字典
        data_names = {
            'head_image': '头部图像',
            'left_hand_image': '左手图像',
            'right_hand_image': '右手图像',
            'joint_state': '关节状态',
            'left_gripper':  '左夹爪状态',
            'right_gripper': '右夹爪状态',
        }
        #遍历self.data_ready字典，检查哪些数据尚未就绪
        for key, ready in self.data_ready.items():
            if not ready:
                missing.append(data_names[key])
       
        return missing
    
    #获取并打包当前观测数据（图像、关节状态、夹爪状态和任务描述），发送给推理服务器
    def get_observation(
        self,
        task_description="Pick up the apple from the fruit basket on the right side of the table and place it into the box on the left",
    ):
        if not self.is_data_ready():
            return None
       
        # 处理图像
        video_obs = {}
        # N1.7 checkpoint 的 modality_config 使用完整 key，例如 observation.images.top_head。
        for key in ['top_head', 'hand_left', 'hand_right']:
            if key not in self.latest_images:
                return None
            img = self.latest_images[key]
            resized = cv2.resize(img, (640, 480)) #将图像调整为640x480分辨率
            resized = resized.astype(np.uint8, copy=False)
            correct_key = f"observation.images.{key}"
            video_obs[correct_key] = resized[np.newaxis, np.newaxis, ...] # (B, T, H, W, C)
        
        # 处理关节状态和夹爪状态，打包成符合推理服务器输入格式的字典
        joint_state = np.asarray(self.latest_joint_state, dtype=np.float32)
        if joint_state.shape[0] < 14:
            return None

        # 与 test_0506.py 保持一致：N1.7 checkpoint 的 state key 是左右臂拆分形式。
        left_arm = joint_state[:7]
        right_arm = joint_state[7:14]
        state_obs = {
            'left_arm_joint_position': left_arm[np.newaxis, np.newaxis, :],
            'right_arm_joint_position': right_arm[np.newaxis, np.newaxis, :],
            'left_effector_position': np.array([[[self.latest_left_gripper]]], dtype=np.float32),
            'right_effector_position': np.array([[[self.latest_right_gripper]]], dtype=np.float32),
        }
       
        language_obs = {
            'annotation.human.task_description': [[task_description]]
        }
       
        return {
            'video': video_obs,
            'state': state_obs,
            'language': language_obs,
        }
   
    def execute_action_step(self, joint_pos, left_gripper, right_gripper):
        """执行单步动作"""
        if not self.enable_control:
            return
       
        # 1. 发布手臂控制
        arm_msg = JointState() #创建消息对象
        arm_msg.header.stamp = self.get_clock().now().to_msg() #填充时间戳
        arm_msg.position = joint_pos.tolist() #将numpy数组转换为列表，填充位置数据
        self.arm_pub.publish(arm_msg) #发布手臂控制消息
       
        # 2. 发布左夹爪控制
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = self.get_clock().now().to_msg()
        left_gripper_msg.name = ['left_gripper_joint1']
        left_gripper_msg.position = [float(left_gripper)]
        self.left_gripper_pub.publish(left_gripper_msg)
       
        # 3. 发布右夹爪控制
        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = self.get_clock().now().to_msg()
        right_gripper_msg.name = ['right_gripper_joint1']
        right_gripper_msg.position = [float(right_gripper)]
        self.right_gripper_pub.publish(right_gripper_msg)
   
    # def execute_action_sequence(self, action, control_freq=5):
    #     """执行完整的32步动作序列"""
    #     if not self. enable_control:
    #         return
       
    #     num_steps = action['left_arm_joint_position'].shape[1]
    #     dt = 1.0 / control_freq
       
    #     print(f"\n🎬 开始执行 {num_steps} 步动作序列 (频率: {control_freq}Hz, 每步{dt*1000:.0f}ms)")
       
    #     for step in range(num_steps):
    #         left_arm = action['left_arm_joint_position'][0, step, :]
    #         right_arm = action['right_arm_joint_position'][0, step, :]
    #         left_gripper = action['left_effector_position'][0, step, 0]
    #         right_gripper = action['right_effector_position'][0, step, 0]
           
    #         self. execute_action_step(left_arm, right_arm, left_gripper, right_gripper)
           
    #         if step % 10 == 0 or step == num_steps - 1:
    #             print(f"  步骤 {step+1:2d}/{num_steps}:  左臂J0={left_arm[0]:+.4f}, 右臂J0={right_arm[0]:+.4f}")
           
    #         time.sleep(dt)
    #         rclpy.spin_once(self, timeout_sec=0.001)
       
    #     print(f"✅ 动作序列发送完成")
        
    #     # ⭐⭐⭐ 新增：强制刷新状态
    #     print(f"🔄 刷新机器人状态...")
    #     for i in range(20):  # 刷新20次，每次50ms，共1秒
    #         rclpy. spin_once(self, timeout_sec=0.05)
    #         time.sleep(0.05)
   
    #     print(f"   当前状态: 左J0={self.latest_joint_state[0]:+.4f}, 右J0={self.latest_joint_state[7]:+.4f}")

    def execute_action_sequence(self, action, control_freq=5, interpolation_factor=5):
        if not self.enable_control:
            return

        original_num_steps = action['joint_position'].shape[1] #获取模型返回的初始步数 （batch, time_step, joint_dim）
        new_num_steps = original_num_steps * interpolation_factor

        dt = 1.0 / control_freq #控制步数周期，如果 control_freq = 5Hz，则 dt = 0.2秒（每步200ms）

        # === 1. 提取第一个 batch 的动作，保持 shape: (T, J) ===
        #left_arm_orig = action['left_arm_joint_position'][0]
        #right_arm_orig = action['right_arm_joint_position'][0]
        joint_orig = action['joint_position'][0]  # (T, 14)
        left_gripper_orig = action['left_effector_position'][0, :, 0]   # 取batch=0的所有步，去掉最后一个维度；(T,)
        right_gripper_orig = action['right_effector_position'][0, :, 0] # (T,) 

        # === 2. 对关节位置做样条插值 ===
        t_orig = np.linspace(0, 1, original_num_steps) #创建从0-1的等差数列，长度为模型返回的动作步数
        t_new = np.linspace(0, 1, new_num_steps) #创建插值后的时间点

        # 定义样条插值函数。N1.7 的 action horizon 不一定是 32，所以阶数按步数自适应。
        def interpolate_spline(orig_traj, t_orig, t_new):
            # orig_traj: (N, D), N=time steps, D=DOF
            N, D = orig_traj.shape #(32,14)
            interp_traj = np.zeros((len(t_new), D)) #创建空数组存放插值结果
            if N == 1:
                return np.repeat(orig_traj, len(t_new), axis=0)
            spline_degree = min(5, N - 1)
            for d in range(D):#对每个关节单独做插值
                # 每个关节：y = orig_traj[:, d] -> shape (N,)
                spl = make_interp_spline(t_orig, orig_traj[:, d], k=spline_degree) #创建插值函数
                interp_traj[:, d] = spl(t_new)#在新时间点计算插值后的关节位置
            return interp_traj  # (new_N, D)

        #left_arm_interp = interpolate_quintic(left_arm_orig, t_orig, t_new)      # (96, J)
        #right_arm_interp = interpolate_quintic(right_arm_orig, t_orig, t_new)    # (96, J)
        
        joint_interp = interpolate_spline(joint_orig, t_orig, t_new)    # (new_N, 14)
        # === 3. 夹爪：零阶保持（每个原始值重复5次）===
        left_gripper_interp = np.repeat(left_gripper_orig, interpolation_factor)   # (new_N,)
        right_gripper_interp = np.repeat(right_gripper_orig, interpolation_factor) # (new_N,)

        # === 4. 执行插值后的动作序列 ===
        num_steps = new_num_steps
        print(
            f"\n🎬 开始执行动作序列: 原始{original_num_steps}步 → 插值{num_steps}步 "
            f"(频率: {control_freq}Hz, 每步{dt*1000:.0f}ms)"
        )

        for step in range(num_steps):
            #left_arm = left_arm_interp[step]
            #right_arm = right_arm_interp[step]
            joint_pos = joint_interp[step]
            left_gripper = left_gripper_interp[step]
            right_gripper = right_gripper_interp[step]

            self.execute_action_step(joint_pos, left_gripper, right_gripper) #逐步发送插值后的动作给机器人
            if step % 30 == 0 or step == num_steps - 1:
                print(f"  步骤 {step+1:2d}/{num_steps}:  关节={format_vector(joint_pos)}")

            time.sleep(dt)
            rclpy.spin_once(self, timeout_sec=0.001)

        print(f"✅ 动作序列发送完成")

        #刷新状态（获取最新的关节状态）
        print(f"🔄 刷新机器人状态...")
        for i in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        current_joints = np.asarray(self.latest_joint_state, dtype=np.float32)
        print(f"   当前状态: 左J0={current_joints[0]:+.4f}, 右J0={current_joints[7]:+.4f}")
    
    #发布话题，等待到达目标位置
    def wait_for_target_position(self, target_action, position_tolerance=0.03, gripper_tolerance=120.0,
                              expected_time=15.0, check_interval=0.1, enable_timeout_warning=False):

        if not self.enable_control:
            return True
   
    # 提取目标位置（最后一步）
        #target_left_arm = target_action['left_arm_joint_position'][0, -1, :]
        #target_right_arm = target_action['right_arm_joint_position'][0, -1, :]
        target_joints = target_action['joint_position'][0, -1, :] #所有关节的最后状态
        target_left_gripper = target_action['left_effector_position'][0, -1, 0] #最后一步的左夹爪
        target_right_gripper = target_action['right_effector_position'][0, -1, 0] #最后一步的右夹爪
   
        print(f"⏳ 等待到达目标...", end='', flush=True)
   
        start_time = time.time()
        timeout_warned = False
   
        while True:  # 无限循环，直到到达目标
            rclpy.spin_once(self, timeout_sec=0.01) #刷新，获取最新数据
       
            if self.latest_joint_state is None:
                time.sleep(check_interval)
                continue 
       
            current_joints = np.array(self.latest_joint_state)
       
            if len(current_joints) < 14:
                time.sleep(check_interval)
                continue
       
        # 关节误差
            joint_errors = np.abs(current_joints - target_joints) # 计算当前关节角度与目标关节角度的误差
            max_joint_error = np.max(joint_errors) # 取最大误差
       
        # 夹爪误差，如果有夹爪数据，计算误差，否则设为999（表示无效）
            left_gripper_error = abs(self.latest_left_gripper - target_left_gripper) if self.latest_left_gripper is not None else 999
            right_gripper_error = abs(self.latest_right_gripper - target_right_gripper) if self.latest_right_gripper is not None else 999
       
            elapsed = time.time() - start_time # 计算到达目标位置用的时间
       
        # 如果误差小于阈值，则返回True
            if (max_joint_error < position_tolerance and
                left_gripper_error < gripper_tolerance and
                right_gripper_error < gripper_tolerance):
           
            #到达目标位置打印
                print(f" ✅ 确认到达目标位置 (耗时: {elapsed:.2f}秒)")
                return True
       
            time.sleep(check_interval)

#订阅话题数据后，确认数据就绪
def wait_for_data(node, timeout=30):
    """等待所有必需数据就绪"""
    print(f"\n⏳ 等待所有必需数据 (超时: {timeout}秒)...")
   
    start_time = time. time()
    last_print_time = 0
   
    while time.time() - start_time < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
       
        if node.is_data_ready():
            print(f"\n✅ 所有数据就绪!  (耗时: {time. time() - start_time:.1f}秒)")
            return True
       
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            elapsed = current_time - start_time
            current_status = node.get_data_status()
            print(f"  [{elapsed:.1f}s] {current_status}") #每秒打印一次节点状态
            last_print_time = current_time
   
    print(f"\n❌ 超时!  未接收到完整数据")
    print(f"   最终状态: {node.get_data_status()}")
   
    missing = node.get_missing_data()
    if missing:
        print(f"   ❌ 缺失数据: {', '.join(missing)}")
   
    return False


def save_observation_images(obs, save_dir="observation_samples"):
    """保存观测图像"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    saved_files = []
    for key, value in obs['video'].items():
        img = value[0, 0, : , :, :] #第一个batch的第一个时间步的图像
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        camera_name = str(key).replace("/", "_").replace(".", "_")
        filename = f"{save_dir}/{timestamp}_{camera_name}.png"
        cv2.imwrite(filename, img_bgr)
        saved_files.append(filename)
   
    return saved_files


def main():
    """主函数"""
    print("=" * 80)
    print("AGIBOT G1推理测试")
    print("=" * 80)
   
    # 控制参数
    ENABLE_CONTROL = False          # 只看模型输出，不向机器人下发动作
    MAX_INFERENCE_CYCLES = 1        # 观察模式默认只跑一次推理后退出
    CONTROL_FREQ = 20              # 20Hz控制频率
    ACTION_INTERPOLATION_FACTOR = 5 # 将模型原始动作序列插值放大，降低真实机器人动作突变
    POSITION_TOLERANCE = 0.03     # 位置容差:  0.03 rad ≈ 1.7°
    EXPECTED_TIME = 15.0          # 预期到达时间（超时只警告）
    ENABLE_TIMEOUT_WARNING = True # 是否启用超时警告
   
    if ENABLE_CONTROL:
        print("\n 控制已启用!  机器人将执行推理动作!")
        print(f"   控制频率:  {CONTROL_FREQ}Hz (每步{1000/CONTROL_FREQ:.0f}ms)")
        print(f"   动作插值倍数: {ACTION_INTERPOLATION_FACTOR}x (原始步数由 N1.7 服务端返回)")
        print(f"   位置容差: {POSITION_TOLERANCE:.4f} rad ({np.degrees(POSITION_TOLERANCE):.2f}°)")
        if ENABLE_TIMEOUT_WARNING:
            print(f"   预期到达时间: {EXPECTED_TIME}秒 (超时只警告，不停止)")
        else:
            print(f"   无超时限制 (会一直等到到达)")
        print("   请确保机器人周围安全!")
        response = input("\n   继续?  (yes/no): ")
        if response.lower() != 'yes':
            print("已取消")
            return
    else:
        print("\n🔒 只观察模型输出模式：不会创建控制发布者，也不会下发动作。")
        print(f"   推理次数: {MAX_INFERENCE_CYCLES}")
   
    # 初始化ROS2
    print("\n[1/3] 初始化ROS2...")
    rclpy.init()
    node = RobotObservationNode(enable_control=ENABLE_CONTROL)
   
    # 连接推理服务器
    print("\n[2/3] 连接推理服务器...")
    try:
        client = PolicyClient(host="127.0.0.1", port=5555)
        print("✅ 连接成功!")
        try:
            client.get_modality_config()
            print("   ✅ Server modality_config 获取成功")
        except Exception as e:
            print(f"   ⚠️ 获取 modality_config 失败（不影响 get_action）: {e}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        rclpy.shutdown()
        return
   
    # 等待数据就绪
    print("\n[3/3] 等待所有必需数据...")
    if not wait_for_data(node, timeout=30):
        print("\n💡 调试提示:")
        print("   1. 检查机器人是否启动")
        print("   2. 检查copilot模式")
        rclpy.shutdown()
        return
   
    # 显示初始状态
    obs = node.get_observation()
    if obs:
        print("\n📊 初始状态:")
        print(f"   关节: {format_vector(node.latest_joint_state)} rad")
        save_observation_images(obs)
        
   
    # 推理循环
    print("\n" + "=" * 80)
    print("开始推理输出检查循环")
    print("   流程:  推理 → 打印模型输出 → 退出")
    print("   按 Ctrl+C 退出")
    print("=" * 80)
   
    try:
        count = 0
       
        while rclpy.ok():
            cycle_start = time.time()
           
            # 1. 获取观测（当前最新状态）
            rclpy.spin_once(node, timeout_sec=0.001)
            obs = node.get_observation()
           
            if obs is None:
                continue
           
            count += 1
           
            print(f"\n\n{'#'*80}")
            print(f"# 推理循环 #{count}")
            print(f"{'#'*80}")
           
            # 记录推理前的状态
            pre_inference_joints = np.array(node.latest_joint_state)
            print(f"\n📍 推理前状态:")
            print(f"   当前关节: {format_vector(pre_inference_joints)} rad")

           
            # 2. 推理
            print(f"\n🧠 正在推理...")
            inference_start = time.time()
            raw_action, _ = client.get_action(obs)
            action = adapt_action_for_control(raw_action)
            inference_time = time. time() - inference_start
            print(f"✅ 推理完成 (耗时: {inference_time*1000:.1f}ms)")
            print_model_output(raw_action, action, pre_inference_joints)
           
            # 显示目标位置（最后一步）
            target_joints = action['joint_position'][0, -1, :]
            action_horizon = action['joint_position'].shape[1]
            print(f"\n🎯 目标位置 (第{action_horizon}步/最后步):")
            print(
                f"   关节: {format_vector(target_joints)} rad "
                f"(变化: {format_vector(target_joints - pre_inference_joints)})"
            )

           
            # 3. 执行动作序列
            if ENABLE_CONTROL:

                # 提取当前 action 的第一个动作
                #curr_first_left = action['left_arm_joint_position'][0, 0, :]   # (7,)
                #curr_first_right = action['right_arm_joint_position'][0, 0, :]
                curr_joints_first =action['joint_position'][0, 0, :]  # (14,)

                # 如果存在上一个动作的结束状态，则插入14步过渡
                if (node.last_joint_pos is not None and
                    node.last_left_gripper is not None and
                    node.last_right_gripper is not None):

                    dt = 1.0 / CONTROL_FREQ
                    num_interp_steps = 14
                    t_new = np.linspace(0, 1, num_interp_steps + 2)[1:-1]  # (4,) → [0.2, 0.4, 0.6, 0.8]

                    def quintic_blend(p0, p1, t_vals):
                        p0, p1, t = np.asarray(p0), np.asarray(p1), np.asarray(t_vals)
                        s = 10 * t**3 - 15 * t**4 + 6 * t**5
                        return p0 + (p1 - p0) * s[:, np.newaxis]  # (N, D)

                    joint_interp = quintic_blend(node.last_joint_pos, curr_joints_first, t_new)      
                    #right_interp = quintic_blend(node.last_right_arm, curr_first_right, t_new)  # (4, 7)

                    # 夹爪保持上一状态
                    left_gripper_vals = np.full(num_interp_steps, node.last_left_gripper)
                    right_gripper_vals = np.full(num_interp_steps, node.last_right_gripper)

                    # 执行过渡步
                    for i in range(num_interp_steps):
                        node.execute_action_step(
                            joint_interp[i],
                            left_gripper_vals[i],
                            right_gripper_vals[i]
                        )
                        time.sleep(dt)
                        rclpy.spin_once(node, timeout_sec=0.001)

                # === 现在执行主动作序列 ===
                node.execute_action_sequence(
                    action,
                    control_freq=CONTROL_FREQ,
                    interpolation_factor=ACTION_INTERPOLATION_FACTOR,
                )

                # === 更新 last state 为当前 action 的最后一个原始动作 ===
                node.last_joint_pos = np.array(action['joint_position'][0, -1, :])
                node.last_left_gripper = float(action['left_effector_position'][0, -1, 0])
                node.last_right_gripper = float(action['right_effector_position'][0, -1, 0])
               
                # 4. 等待到达目标位置（会一直等到到达）
                node.wait_for_target_position(
                    action,
                    position_tolerance=POSITION_TOLERANCE,
                    expected_time=EXPECTED_TIME,
                    enable_timeout_warning=ENABLE_TIMEOUT_WARNING
                )
               
                # 5. 到达后，验证最终位置
                rclpy.spin_once(node, timeout_sec=0.01)
                final_joints = np.array(node.latest_joint_state)
               
                print(f"\n📊 执行结果:")
                joint_error = np.abs(final_joints - target_joints)
                print(f"   目标: 关节{format_vector(target_joints)}")
                print(f"   实际: 关节{format_vector(final_joints)}")
                print(
                    f"   误差: 关节{format_vector(joint_error)} rad "
                    f"(max={np.max(joint_error):.4f})"
                )
               
            else:
                print("  ⏭️  跳过执行 (控制未启用)")
           
            # 统计
            cycle_time = time.time() - cycle_start
            print(f"\n⏱️  本轮耗时: {cycle_time:.2f}秒")

            if not ENABLE_CONTROL and count >= MAX_INFERENCE_CYCLES:
                print("\n✅ 模型输出检查完成，已退出。")
                break
           
            # ⭐ 到达后立即进入下一轮推理，不额外等待
   
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    finally:
        rclpy.shutdown()
        print("🔌 已关闭ROS2节点")


if __name__ == "__main__":
    main()
