#pragma once

#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>
#include <chrono>
#include <algorithm>
#include <btBulletCollisionCommon.h>
#include <thread>
#include <chrono>
#include "RobotSimulator.h"
#include <tinyxml/tinyxml2.h>
#include <sys/stat.h>
#include "Manipulator.h"
// Include individual environment headers

// Abstract environment interface
class Env {
public:
    torch::Device& mDevice;
    Env(torch::Device& device) : mDevice(device) {}

    virtual ~Env() = default;

    struct Space {
        std::vector<int> shape;
    };

    virtual torch::Tensor reset(int index = -1) = 0;
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions, int frame_index) = 0;
    virtual void render() = 0;
    virtual void animate() = 0;
    virtual Space observation_space() const = 0;
    virtual Space action_space() const = 0;
    virtual void EnableManipulator() = 0;
    int GetGridCount() const { return (grid_size * grid_size); }
    int grid_size = 1;
    float grid_space = 20.0f;
};

class Env3D : public Env {
public:
    Env3D(torch::Device& device, b3RobotSimulatorClientAPI* sim_ptr) : Env(device), sim(sim_ptr)
    {
        if (!sim->connect(eCONNECT_GUI)) {
            printf("Cannot connect\n");
            return;
        }

        sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
        sim->setTimeOut(10);
        sim->syncBodies();
        sim->setTimeStep(1. / 240.);
        sim->setGravity(btVector3(0, 0, -9.8));

        // Load animation XML once
        LoadAnimationXML();
    }
    virtual ~Env3D() override = default;

    b3RobotSimulatorClientAPI* sim;
    std::vector<int> agent_ids;
    std::vector<int> target_ids;
    /// ANIMATION CODE
    int selected_joint_index = 0; // Track selected joint
    int selected_object_id = 0; // Track selected humanoid
    int current_animation_frame = 0; // Track current animation frame

    btVector3 start_pos = btVector3(0, 0, 0);
    btQuaternion start_ori = btQuaternion(0, 0, 0, 1);

    tinyxml2::XMLDocument animation_doc;

    Manipulator* manipulator;
 
    void EnableManipulator() override
    {
        HINSTANCE hInstance = GetModuleHandle(NULL);
        int nCmdShow = SW_SHOWNORMAL;

        try {
            if (!sim || agent_ids.empty()) {
                MessageBoxA(NULL, "Simulator not initialized or no agents loaded. Cannot enable manipulator.", "Error", MB_ICONERROR);
                return;
            }

            int object_id = agent_ids[0]; // Assuming agent_ids[0] holds the unique ID of the robot

            int numJoints = sim->getNumJoints(object_id);

            if (numJoints <= 0) {
                MessageBoxA(NULL, "No joints found for the agent. Manipulator not enabled.", "Info", MB_OK | MB_ICONINFORMATION);
                return;
            }

            manipulator = new Manipulator(hInstance, 100, 100, 400, 150 + numJoints * 40, "Robot Arm Control", numJoints);
            manipulator->Show(nCmdShow);

            // Declare joint_positions here, so it's a mutable variable that can be captured by reference
            std::vector<double> joint_positions(numJoints);

            for (int j = 0; j < numJoints; ++j) {
                b3JointInfo jointInfo;
                sim->getJointInfo(object_id, j, &jointInfo);

                const double RAD_TO_DEG = 180.0 / SIMD_PI;

                int minAngle = static_cast<int>(jointInfo.m_jointLowerLimit * RAD_TO_DEG);
                int maxAngle = static_cast<int>(jointInfo.m_jointUpperLimit * RAD_TO_DEG);

                if (minAngle >= maxAngle) {
                    minAngle = -180;
                    maxAngle = 180;
                }

                manipulator->SetJointRange(j, minAngle, maxAngle);

                // Optional: Initialize joint_positions with current joint states
                b3JointSensorState state;
                sim->getJointState(object_id, j, &state);
                joint_positions[j] = state.m_jointPosition;
            }

            manipulator->SetFrameRange(0, 500);

            manipulator->SetOnManipulatorChangeCallback(
                // Capture 'this' by reference (for sim, agent_ids etc.)
                // Capture joint_positions by reference if it's a member or local.
                // object_id and numJoints are captured here as they are local to EnableManipulator
                [&, object_id, numJoints](const std::vector<int>& positionXYZ,
                    const std::vector<int>& rotationXYZ,
                    const std::vector<int>& jointAngles,
                    int frameNumber) {
                        std::cout << "Position: [" << positionXYZ[0] << " " << positionXYZ[1] << " " << positionXYZ[2] << "], ";
                        std::cout << "Rotation: [" << rotationXYZ[0] << " " << rotationXYZ[1] << " " << rotationXYZ[2] << "], ";
                        std::cout << "Joint Angles: [";
                        for (int angle : jointAngles) {
                            std::cout << angle << " ";
                        }
                        std::cout << "], Frame: " << frameNumber << std::endl;

                        // --- Apply Base Position and Orientation ---
                        // Convert integer slider values (e.g., in degrees/units) to Bullet's required types (doubles/quaternions)
                        // Assuming positionXYZ values are directly usable as coordinates (e.g., meters)
                        btVector3 start_pos(static_cast<double>(positionXYZ[0]),
                            static_cast<double>(positionXYZ[1]),
                            static_cast<double>(positionXYZ[2]));

                        // Assuming rotationXYZ are Euler angles in degrees (X, Y, Z - Roll, Pitch, Yaw)
                        // Convert degrees to radians for Bullet's quaternion creation
                        const double DEG_TO_RAD = SIMD_PI / 180.0;
                        btQuaternion start_ori;
                        start_ori.setEulerZYX(static_cast<double>(rotationXYZ[2]) * DEG_TO_RAD, // Z (Yaw)
                            static_cast<double>(rotationXYZ[1]) * DEG_TO_RAD, // Y (Pitch)
                            static_cast<double>(rotationXYZ[0]) * DEG_TO_RAD); // X (Roll)

                        sim->resetBasePositionAndOrientation(object_id, start_pos, start_ori);

                        // --- Apply Joint Angles ---
                        for (int j = 0; j < numJoints; ++j) {
                            if (j < jointAngles.size()) {
                                double targetPositionRad = static_cast<double>(jointAngles[j]) * DEG_TO_RAD;

                                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                                motorArgs.m_targetPosition = targetPositionRad; // Target position in radians
                                motorArgs.m_targetVelocity = 1.0;
                                motorArgs.m_kp = 0.1;
                                motorArgs.m_kd = 0.5;
                                motorArgs.m_maxTorqueValue = 1000.0;

                                sim->setJointMotorControl(object_id, j, motorArgs);
                            }
                        }
                }
            );
        }
        catch (const std::runtime_error& e) {
            MessageBoxA(NULL, e.what(), "Error", MB_ICONERROR);
            return;
        }
    }

    void LoadAnimationXML() {
        std::string xml_path = "animations/animation.xml";
        animation_doc.Clear();
        animation_doc.LoadFile(xml_path.c_str());
    }

    // Remove anim_skip_steps argument from animate
    void animate() override {
        // Mouse interaction for joint manipulation

        static btVector3 last_mouse_pos;

        printf("Animation mode started. Press ESC to exit.\n");
        printf("Press 'S' to save joint positions, 'R' to reset to saved positions.\n");
        printf("Use 1/2 to select joint, 3/4 to modify position.\n");
        printf("Use 5/6 to select animation frame.\n");

        // Run animation loop until escape is pressed
        using clock = std::chrono::steady_clock;
        auto last_event_time = clock::now();
        while (true && agent_ids.size() > 0) {
            // Check for escape key

            auto now = clock::now();
            double elapsed = std::chrono::duration<double>(now - last_event_time).count();
            b3KeyboardEventsData keyboardEvents;
            sim->getKeyboardEvents(&keyboardEvents);
            if (0 < keyboardEvents.m_numKeyboardEvents && elapsed >= 0.1) {
                const b3KeyboardEvent& event = keyboardEvents.m_keyboardEvents[0];

                if (event.m_keyCode == 27 && event.m_keyState == 1) { // ESC key
                    printf("Animation mode exited.\n");
                    return;
                }
                else if (event.m_keyCode == 115 && event.m_keyState == 1) { // 'S' key - Save positions
                    saveJointPositions(current_animation_frame);
                    printf("Joint positions saved.\n");
                }
                else if (event.m_keyCode == 114 && event.m_keyState == 1) { // 'R' key - Reset positions
                    resetJointPositions(current_animation_frame);
                    printf("Joint positions reset to saved state.\n");
                }
                else if (event.m_keyCode == 49 && event.m_keyState == 1) { // '1' key - Previous joint
                    if (agent_ids.empty()) {
                        printf("No humanoids available.\n");
                    }
                    else {
                        int object_id = agent_ids[selected_object_id];
                        int num_joints = sim->getNumJoints(object_id);
                        selected_joint_index = (selected_joint_index - 1 + num_joints) % num_joints;

                        // Get and print the joint name
                        b3JointInfo jointInfo;
                        if (sim->getJointInfo(object_id, selected_joint_index, &jointInfo)) {
                            printf("Selected joint: %d - %s\n", selected_joint_index, jointInfo.m_linkName);
                        }
                        else {
                            printf("Selected joint: %d\n", selected_joint_index);
                        }
                    }
                }
                else if (event.m_keyCode == 50 && event.m_keyState == 1) { // '2' key - Next joint
                    if (agent_ids.empty()) {
                        printf("No humanoids available.\n");
                    }
                    else {
                        int object_id = agent_ids[selected_object_id];
                        int num_joints = sim->getNumJoints(object_id);
                        selected_joint_index = (selected_joint_index + 1) % num_joints;

                        // Get and print the joint name
                        b3JointInfo jointInfo;
                        if (sim->getJointInfo(object_id, selected_joint_index, &jointInfo)) {
                            printf("Selected joint: %d - %s\n", selected_joint_index, jointInfo.m_linkName);
                        }
                        else {
                            printf("Selected joint: %d\n", selected_joint_index);
                        }
                    }
                }
                else if (event.m_keyCode == 51 && event.m_keyState == 1) { // '3' key - Increase position
                    modifySelectedJointPosition(0.1);

                    // Print current joint value
                    if (!agent_ids.empty()) {
                        int object_id = agent_ids[selected_object_id];
                        b3JointSensorState state;
                        if (sim->getJointState(object_id, selected_joint_index, &state)) {
                            printf("Joint %d position: %.3f\n", selected_joint_index, state.m_jointPosition);
                        }
                    }
                }
                else if (event.m_keyCode == 52 && event.m_keyState == 1) { // '4' key - Decrease position
                    modifySelectedJointPosition(-0.1);

                    // Print current joint value
                    if (!agent_ids.empty()) {
                        int object_id = agent_ids[selected_object_id];
                        b3JointSensorState state;
                        if (sim->getJointState(object_id, selected_joint_index, &state)) {
                            printf("Joint %d position: %.3f\n", selected_joint_index, state.m_jointPosition);
                        }
                    }
                }
                else if (event.m_keyCode == 53 && event.m_keyState == 1) { // '5' key - Previous animation frame
                    current_animation_frame = std::max(0, current_animation_frame - 1);
                    printf("Animation frame: %d\n", current_animation_frame);
                }
                else if (event.m_keyCode == 54 && event.m_keyState == 1) { // '6' key - Next animation frame
                    current_animation_frame++;
                    printf("Animation frame: %d\n", current_animation_frame);
                }
                last_event_time = clock::now();
            }

            //sim->resetBasePositionAndOrientation(agent_ids[0], start_pos, start_ori);
            sim->resetBaseVelocity(agent_ids[0], btVector3(0, 0, 0), btVector3(0, 0, 0));

            // Step simulation to apply joint changes
            sim->stepSimulation();

            // Main message loop (if you're not using GraphWindowManager's implicit loop)
            MSG msg = {};
            if(GetMessage(&msg, NULL, 0, 0)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }

    // External function to save joint positions
    void saveJointPositions(int frame = -1) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to save.\n");
                return;
            }
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            std::vector<double> joint_positions;
            for (int j = 0; j < num_joints; ++j) {
                b3JointSensorState state;
                if (sim->getJointState(object_id, j, &state)) {
                    joint_positions.push_back(state.m_jointPosition);
                } else {
                    joint_positions.push_back(0.0);
                }
            }

            // Ensure animations directory exists
            std::filesystem::create_directories("animations");

            // Use cached animation_doc
            tinyxml2::XMLElement* root = animation_doc.FirstChildElement("Animation");
            if (!root) {
                root = animation_doc.NewElement("Animation");
                animation_doc.InsertEndChild(root);
            }

            int frame_index = (frame >= 0) ? frame : current_animation_frame;
            // Find or create frame element
            tinyxml2::XMLElement* frameElem = nullptr;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                if (elem->IntAttribute("index") == frame_index) {
                    frameElem = elem;
                    break;
                }
            }
            if (!frameElem) {
                frameElem = animation_doc.NewElement("Frame");
                frameElem->SetAttribute("index", frame_index);
                root->InsertEndChild(frameElem);
            } else {
                frameElem->DeleteChildren(); // Clear previous joint data
            }

            // Save joint positions
            for (int j = 0; j < num_joints; ++j) {
                tinyxml2::XMLElement* jointElem = animation_doc.NewElement("Joint");
                jointElem->SetAttribute("index", j);
                jointElem->SetAttribute("position", joint_positions[j]);
                frameElem->InsertEndChild(jointElem);
            }

            // --- Sort <Frame> elements by 'index' attribute before saving ---
            std::vector<tinyxml2::XMLElement*> frames;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                frames.push_back(elem);
            }
            std::sort(frames.begin(), frames.end(), [](tinyxml2::XMLElement* a, tinyxml2::XMLElement* b) {
                return a->IntAttribute("index") < b->IntAttribute("index");
            });
            for (tinyxml2::XMLElement* elem : frames) {
                root->InsertEndChild(elem);
            }
            // --- End sorting ---

            animation_doc.SaveFile("animations/animation.xml");
        } catch (const std::exception& e) {
            printf("Exception in saveJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in saveJointPositions.\n");
        }
    }

    // External function to reset joint positions
    void resetJointPositions(int frame = -1) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to reset.\n");
                return;
            }
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            std::vector<double> joint_positions = GetJointAnim(frame);
            for (int j = 0; j < num_joints; ++j) {
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = joint_positions[j];
                motorArgs.m_targetVelocity = 1.0; // Large value for instant movement
                motorArgs.m_maxTorqueValue = 100.0; // Large torque for instant movement
                sim->setJointMotorControl(object_id, j, motorArgs);
            }
        } catch (const std::exception& e) {
            printf("Exception in resetJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in resetJointPositions.\n");
        }
    }

    void modifySelectedJointPosition(double delta) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available.\n");
                return;
            }
            if (selected_object_id >= agent_ids.size()) {
                selected_object_id = 0;
            }
            int object_id = agent_ids[selected_object_id];
            int num_joints = sim->getNumJoints(object_id);
            if (selected_joint_index >= num_joints) {
                selected_joint_index = 0;
            }

            // Get current joint position
            b3JointSensorState state;
            if (sim->getJointState(object_id, selected_joint_index, &state)) {
                double new_position = state.m_jointPosition + delta;

                // Apply the new position
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = new_position;
                motorArgs.m_targetVelocity = 1.0;
                motorArgs.m_maxTorqueValue = 100.0;
                sim->setJointMotorControl(object_id, selected_joint_index, motorArgs);

                printf("Joint %d position: %.3f\n", selected_joint_index, new_position);
            }
        }
        catch (const std::exception& e) {
            printf("Exception in modifySelectedJointPosition: %s\n", e.what());
        }
        catch (...) {
            printf("Unknown exception in modifySelectedJointPosition.\n");
        }
    }

    // Get joint animation for the current or last frame
    std::vector<double> GetJointAnim(int frame = -1) {
        std::vector<double> joint_positions;
        tinyxml2::XMLElement* animation_root = animation_doc.FirstChildElement("Animation");
        int num_joints = 0;
        if (!agent_ids.empty()) {
            int object_id = agent_ids[0];
            num_joints = sim->getNumJoints(object_id);
        }
        if (animation_doc.ErrorID() != 0 || !animation_root) {
            // No animation file: return zeros
            return std::vector<double>(num_joints, 0.0);
        }
        // Find the frame with the given frame number, or last if frame == -1
        tinyxml2::XMLElement* frameElem = nullptr;
        int last_index = -1;
        tinyxml2::XMLElement* lastElem = nullptr;
        for (tinyxml2::XMLElement* elem = animation_root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
            int idx = elem->IntAttribute("index");
            if ((frame >= 0 && idx == frame)) {
                frameElem = elem;
                break;
            }
            if (idx > last_index) {
                last_index = idx;
                lastElem = elem;
            }
        }
        if (!frameElem && lastElem) frameElem = lastElem;
        if (!frameElem) return std::vector<double>(num_joints, 0.0);
        // Read joint positions
        for (tinyxml2::XMLElement* jointElem = frameElem->FirstChildElement("Joint"); jointElem; jointElem = jointElem->NextSiblingElement("Joint")) {
            joint_positions.push_back(jointElem->DoubleAttribute("position"));
        }
        // If mismatch, return zeros
        if (joint_positions.size() != static_cast<size_t>(num_joints)) {
            return std::vector<double>(num_joints, 0.0);
        }
        return joint_positions;
    }
};
