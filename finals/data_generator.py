import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pickle  # For saving data
import threading  # For non-blocking input
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin


print_cartesian_trajectory = False
def main():
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Get joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")

    # Controller gains
    kp_pos = 100  # Position gain
    kp_ori = 0    # Orientation gain
    kp = 1000     # Proportional gain for feedback linearization
    kd = 100      # Derivative gain for feedback linearization

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    time_all, goal_positions_all = [], []

    # Time management
    time_step = sim.GetTimeStep()
    current_time = 0

    # Randomness control
    random_seed = 42  # Set a seed for reproducibility
    np.random.seed(random_seed)

    # Goal position bounds (example values, adjust as needed)
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }

    # Trajectory parameters
    total_trajectories = 5  # Number of trajectories to execute
    trajectory_duration = 5.0  # Duration of each trajectory in seconds

    # Initialize command structure for motors
    cmd = MotorCommands()

    # Loop over multiple trajectories
    for trajectory_idx in range(total_trajectories):
        print(f"\nStarting trajectory {trajectory_idx + 1}/{total_trajectories}")

        # Generate a random goal position within specified bounds
        goal_position = np.array([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])
        print(f"Generated goal position: {goal_position}")

        #  recompute cartesian position
        init_init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
        # Create a spline trajectory from the initial to the goal position
        ref = CartesianSplineReference(init_init_cartesian_pos, goal_position, trajectory_duration)

        # plotting each ref to check if it is working
        time_array = np.linspace(0, trajectory_duration, num=100)
        ref_positions = [ref.get_values(t)[0] for t in time_array]
        # print the initial value of the trajectory
        print(f"Initial position of the trajectory: {init_init_cartesian_pos}")
        # print the final value of the trajectory
        print(f"Final position of the trajectory: {ref_positions[-1]}")

        if print_cartesian_trajectory:
            plot_reference_trajectory(time_array, ref_positions)
        
        # Reset the robot to the initial position
        sim.ResetPose()
        current_time = 0  # Reset time for the new trajectory

        # Control loop for the trajectory
        while current_time <= trajectory_duration:
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

            # printing the current end effector position
            cur_ee = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            print(f"Current end effector position: {cur_ee[0]}")

            # Get desired position and velocity from the spline
            p_d, pd_d = ref.get_values(current_time)  # Desired position and velocity

            # Inverse differential kinematics
            ori_des = None
            ori_d_des = None
            q_des, qd_des_clip = CartesianDiffKin(
                dyn_model, controlled_frame_name, q_mes, p_d, pd_d,
                ori_des, ori_d_des, time_step, "pos", kp_pos, kp_ori, np.array(joint_vel_limits)
            )

            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * num_joints)
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Exit logic with 'q' key
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                return

            # Store data
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)
            q_d_all.append(q_des)
            qd_d_all.append(qd_des_clip)
            time_all.append(current_time)
            goal_positions_all.append(goal_position)

            # Time management
            time.sleep(time_step)
            current_time += time_step
            print(f"Current time in seconds: {current_time:.2f}")


    # After all trajectories, prompt for downsample rate and save data
    downsample_and_save_data(
        time_all, goal_positions_all, q_mes_all, qd_mes_all, q_d_all, qd_d_all
    )

def downsample_and_save_data(time_all, goal_positions_all, q_mes_all, qd_mes_all, q_d_all, qd_d_all):
    # Function to get downsample rate from the user without blocking
    def get_downsample_rate():
        try:
            rate = int(input("Enter downsample rate (integer >=1): "))
            if rate < 1:
                print("Invalid downsample rate. Must be >= 1.")
                return None
            return rate
        except ValueError:
            print("Please enter a valid integer.")
            return None

    print("Simulation completed. Preparing to save data...")
    downsample_rate = get_downsample_rate()
    

    # Downsample data
    time_downsampled = time_all[::downsample_rate]
    goal_positions_downsampled = goal_positions_all[::downsample_rate]
    q_mes_all_downsampled = q_mes_all[::downsample_rate]

    # Save data to pickle file
    filename = "data.pkl"
    with open(filename, 'wb') as f:
        pickle.dump({
            'time': time_downsampled,
            'goal_positions': goal_positions_downsampled,
            'q_mes_all': q_mes_all_downsampled,
        }, f)
    print(f"Data saved to {filename}")

    # Optional: Plot joint positions
    plot_joint_positions(time_downsampled, q_mes_all_downsampled)

def plot_reference_trajectory(time_array, ref_positions):
    # Extract x, y, z components
    x_positions = [pos[0] for pos in ref_positions]
    y_positions = [pos[1] for pos in ref_positions]
    z_positions = [pos[2] for pos in ref_positions]

    # Plot x, y, z positions over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_array, x_positions, label='X Position')
    plt.plot(time_array, y_positions, label='Y Position')
    plt.plot(time_array, z_positions, label='Z Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Reference Trajectory Positions Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_joint_positions(time_array, q_mes_all):
    import matplotlib.pyplot as plt

    # Plot joint positions
    plt.figure(figsize=(12, 6))
    for joint_idx in range(len(q_mes_all[0])):
        joint_positions = [q[joint_idx] for q in q_mes_all]
        plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Positions (rad)')
    plt.title('Joint Positions Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

class CartesianSplineReference:
    def __init__(self, start_pos, goal_pos, duration):
        self.start_pos = start_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.duration = duration
        self.coefficients = self.compute_spline_coefficients()

    def compute_spline_coefficients(self):
        # Cubic polynomial coefficients for each axis
        # Assuming zero initial and final velocities
        coeffs = []
        for i in range(len(self.start_pos)):
            a0 = self.start_pos[i]
            a1 = 0
            a2 = (3 / self.duration**2) * (self.goal_pos[i] - self.start_pos[i])
            a3 = (-2 / self.duration**3) * (self.goal_pos[i] - self.start_pos[i])
            coeffs.append((a0, a1, a2, a3))
        return coeffs

    def get_values(self, t):
        # Compute position and velocity at time t
        pos = []
        vel = []
        for coeff in self.coefficients:
            a0, a1, a2, a3 = coeff
            p = a0 + a1 * t + a2 * t**2 + a3 * t**3
            v = a1 + 2 * a2 * t + 3 * a3 * t**2
            pos.append(p)
            vel.append(v)
        return np.array(pos), np.array(vel)

if __name__ == "__main__":
    main()
