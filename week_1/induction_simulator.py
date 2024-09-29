import numpy as np
import time
import os
import simulation_and_control as sac
from simulation_and_control.sim import pybullet_robot_interface as pb
from simulation_and_control.controllers.servo_motor import MotorCommands
from simulation_and_control.controllers.pin_wrapper import PinWrapper

def main():
    # Configuration for the simulation
    conf_file_name = "elephantconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)

    # Display dynamics information
    print("Joint info simulator:")
    sim.GetBotJointsInfo()

    print("Link info simulator:")
    sim.GetBotDynamicsInfo()

    print("Link info pinocchio:")
    dyn_model.getDynamicsInfo()

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    while True:
        cmd.tau_cmd = np.zeros((dyn_model.getNumberofActuatedJoints(),))  # Zero torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)): # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        time.sleep(0.1)  # Slow down the loop for better visualization

if __name__ == '__main__':
    main()