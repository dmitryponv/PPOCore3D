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
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) = 0;
    virtual void render() = 0;
    virtual void animate(int anim_skip_steps = 1) = 0;
    virtual Space observation_space() const = 0;
    virtual Space action_space() const = 0;
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
    }
    virtual ~Env3D() override = default;

    b3RobotSimulatorClientAPI* sim;
    std::vector<int> agent_ids;
    std::vector<int> target_ids;
    /// ANIMATION CODE
    std::vector<double> saved_joint_positions; // Store joint positions
    int selected_joint_index = 0; // Track selected joint
    int selected_object_id = 0; // Track selected humanoid
    int current_animation_frame = 0; // Track current animation frame

    btVector3 start_pos = btVector3(0, 0, 0);
    btQuaternion start_ori = btQuaternion(0, 0, 0, 1);

    void animate(int anim_skip_steps = 1) override {
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
                    saveJointPositions();
                    printf("Joint positions saved.\n");
                }
                else if (event.m_keyCode == 114 && event.m_keyState == 1) { // 'R' key - Reset positions
                    resetJointPositions();
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

            // Freeze base position and velocity for all loaded URDF humanoids
            int i = 0; // Use first grid position for simplicity
            int j = 0;
            sim->resetBasePositionAndOrientation(agent_ids[0], start_pos, start_ori);
            sim->resetBaseVelocity(agent_ids[0], btVector3(0, 0, 0), btVector3(0, 0, 0));

            // Step simulation to apply joint changes
            sim->stepSimulation();

            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }

    // External function to save joint positions
    void saveJointPositions() {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to save.\n");
                return;
            }
            saved_joint_positions.clear();
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            for (int j = 0; j < num_joints; ++j) {
                b3JointSensorState state;
                if (sim->getJointState(object_id, j, &state)) {
                    saved_joint_positions.push_back(state.m_jointPosition);
                }
                else {
                    saved_joint_positions.push_back(0.0);
                }
            }
        }
        catch (const std::exception& e) {
            printf("Exception in saveJointPositions: %s\n", e.what());
        }
        catch (...) {
            printf("Unknown exception in saveJointPositions.\n");
        }
    }

    // External function to reset joint positions
    void resetJointPositions() {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to reset.\n");
                return;
            }
            if (saved_joint_positions.empty()) {
                printf("No saved positions to reset to.\n");
                return;
            }
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            for (int j = 0; j < num_joints; ++j) {
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = saved_joint_positions[j];
                motorArgs.m_targetVelocity = 1.0; // Large value for instant movement
                motorArgs.m_maxTorqueValue = 100.0; // Large torque for instant movement
                sim->setJointMotorControl(object_id, j, motorArgs);
            }
        }
        catch (const std::exception& e) {
            printf("Exception in resetJointPositions: %s\n", e.what());
        }
        catch (...) {
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
};
