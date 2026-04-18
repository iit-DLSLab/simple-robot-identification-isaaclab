import readline
import readchar
import time

import numpy as np
import copy

import mujoco

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.setpoint_collection = False
        self.falling_collection = False
        self.trajectory_collection = False
        self.isActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "goDown", "startGeneration", "setKp", "setKd"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                    
                if(input_string == "startCollection"):

                    mode = input("Select the collection mode (setpoint/falling/trajectory): ")
                    if(mode == "setpoint"):
                        self.setpoint_collection = True
                        self.falling_collection = False
                        self.trajectory_collection = False
                        print("Setpoint collection mode activated")
                    elif(mode == "falling"):
                        self.setpoint_collection = False
                        self.falling_collection = True
                        self.trajectory_collection = False
                        print("Falling collection mode activated")
                    elif(mode == "trajectory"):
                        self.setpoint_collection = False
                        self.falling_collection = False
                        self.trajectory_collection = True
                        print("Trajectory collection mode activated")
                    else:
                        print("Invalid mode selected")
                        self.isActivated = False
                        continue
                    
                    self.isActivated = True

                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.Kp_stand_up_and_down = float(temp)
                    


                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.Kd_stand_up_and_down = float(temp)


            
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("goDown: Move the robot down")
        print("startCollection: Start the collection process")
        print("setKp: Set the proportional gain")
        print("setKd: Set the derivative gain\n")