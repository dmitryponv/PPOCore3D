#pragma once
#include "RobotSimulator.h"
#include "env.h"
#include <btBulletCollisionCommon.h>
#include <thread>
#include <chrono>

class HumanoidEnv : public Env {
private:
    b3RobotSimulatorClientAPI* sim;
    std::vector<int> humanoid_ids;
    torch::Device& mDevice;
    // std::vector<std::vector<b3LinkState>> saved_link_states; // Remove unused
    // std::vector<btVector3> saved_base_positions; // Remove unused
    // std::vector<btQuaternion> saved_base_orientations; // Remove unused
    std::vector<double> saved_joint_positions; // Store joint positions
    int selected_joint_index = 0; // Track selected joint
    int selected_humanoid_id = 0; // Track selected humanoid
    int current_animation_frame = 0; // Track current animation frame
public:
    HumanoidEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env(), // Call base class constructor
        mDevice(device)
    {

        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member
        sim = new b3RobotSimulatorClientAPI();
        if (!sim->connect(eCONNECT_GUI)) {
            printf("Cannot connect\n");
            return;
        }

        sim->setGravity(btVector3(0, 0, -9.8));
        sim->setTimeStep(1. / 240.);

        humanoid_ids.clear();

        for (int i = 0; i < this->grid_size; ++i) { // Use this->grid_size
            for (int j = 0; j < this->grid_size; ++j) { // Use this->grid_size
                btVector3 start_pos(i * this->grid_space, j * this->grid_space, 0.0f); // Use this->grid_space

                b3RobotSimulatorLoadUrdfFileArgs plane_args;
                plane_args.m_startPosition = { start_pos.x(), start_pos.y(), start_pos.z() };
                plane_args.m_startOrientation = { 0.0f, 0.0f, 0.0f, 1.0f };
                sim->loadURDF("plane.urdf", plane_args);

                b3RobotSimulatorLoadUrdfFileArgs args;
                args.m_startPosition = { start_pos.x(), start_pos.y(), 1.0f };
                args.m_startOrientation = { 0.0f, 0.0f, 0.0f, 1.0f };
                args.m_useMultiBody = true;
                args.m_flags = 0;

                int id = sim->loadURDF("humanoid.urdf", args);
                humanoid_ids.push_back(id);
            }
        }

        sim->setRealTimeSimulation(false);
    }



    Space observation_space() const override {
        if (humanoid_ids.empty()) return Space{ {0} };
        int num_joints = sim->getNumJoints(humanoid_ids[0]);
        int obs_per_joint = 3 + 4 + 3 + 3; // pos + quat + linear vel + angular vel
        return Space{ {num_joints * obs_per_joint} };
    }

    Space action_space() const override {
        if (humanoid_ids.empty()) return Space{ {0} };
        return Space{ {sim->getNumJoints(humanoid_ids[0])} };
    }

    torch::Tensor reset(int index) override {
        if (index < 0 || index >= humanoid_ids.size()) {
            return torch::empty({ 0 });
        }

        int i = index / grid_size;
        int j = index % grid_size;

        int id = humanoid_ids[index];

        btVector3 start_pos(i * grid_space, j * grid_space, 0.5);
        btQuaternion start_ori;
        start_ori.setEulerZYX(0, M_PI_2, 0); // 90 degrees around Y-axis

        sim->resetBasePositionAndOrientation(id, start_pos, start_ori);
        sim->resetBaseVelocity(id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        int num_joints = sim->getNumJoints(id);
        for (int k = 0; k < num_joints; ++k) {
            sim->resetJointState(id, k, 0.0);
        }

        return get_observation(id);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;

        static b3Clock clock;
        static double lastTime = clock.getTimeInSeconds();
        static int frameCount = 0;
        static int fpsTextId = -1;
        frameCount++;
        double currentTime = clock.getTimeInSeconds();
        double elapsed = currentTime - lastTime;
        if (elapsed >= 1.0) {
            double fps = frameCount / elapsed;
            std::string text = "FPS: " + std::to_string(fps); // Fixed string concatenation
            if (fpsTextId >= 0)
                sim->removeUserDebugItem(fpsTextId);
            double pos[3] = { 0, 0, 2 };
            b3RobotSimulatorAddUserDebugTextArgs args;
            fpsTextId = sim->addUserDebugText(text.c_str(), pos, args);
            lastTime = currentTime;
            frameCount = 0;
        }

        for (size_t i = 0; i < std::min(actions.size(), humanoid_ids.size()); ++i) {
            int id = humanoid_ids[i];
            const torch::Tensor& action = actions[i];
            int num_joints = sim->getNumJoints(id);
            const float max_velocity = 1.0f;
            // bool done = false; // done is determined after sim->stepSimulation()

            for (int j = 0; j < num_joints; ++j) {
                b3JointInfo jointInfo;
                sim->getJointInfo(id, j, &jointInfo);

                if (jointInfo.m_jointType != JointType::eRevoluteType) {
                    continue;
                }

                float action_single = action[j].item<float>() * 10.0f;

                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_maxTorqueValue = 200.0f;
                motorArgs.m_targetPosition = action_single;
                motorArgs.m_targetVelocity = max_velocity;

                sim->setJointMotorControl(id, j, motorArgs);
            }
        }

        sim->stepSimulation();

        for (size_t i = 0; i < humanoid_ids.size(); ++i) {
            int id = humanoid_ids[i];
            int num_joints = sim->getNumJoints(id);

            // Find the link index for "torso_object"
            int torso_link_index = -1;
            for (int j = 0; j < num_joints; ++j) {
                b3JointInfo jointInfo;
                if (sim->getJointInfo(id, j, &jointInfo)) {
                    if (std::string(jointInfo.m_linkName) == "head_object") {
                        torso_link_index = j;
                        break;
                    }
                }
            }

            btVector3 torso_pos(0, 0, 0);
            if (torso_link_index != -1) {
                b3LinkState torso_state;
                sim->getLinkState(id, torso_link_index, 1, 0, &torso_state);
                torso_pos = btVector3(
                    torso_state.m_worldPosition[0],
                    torso_state.m_worldPosition[1],
                    torso_state.m_worldPosition[2]
                );
            }

            btVector3 target_pos_check(0.0f, 0.0f, 2.0f);
            float dist = (torso_pos - target_pos_check).length();
            float reward = dist - 1.5f;
            reward = (torso_pos[2] - target_pos_check[2]);
            bool done = dist > 30;

            results.push_back({ get_observation(id), reward, done, false });
        }

        return results;
    }


    torch::Tensor get_observation(int id) {
        std::vector<float> obs;
        int num_joints = sim->getNumJoints(id);
        for (int j = 0; j < num_joints; ++j) {
            b3LinkState link_state;
            sim->getLinkState(id, j, 1, 0, &link_state);

            obs.push_back(link_state.m_worldPosition[0]);
            obs.push_back(link_state.m_worldPosition[1]);
            obs.push_back(link_state.m_worldPosition[2]);

            obs.push_back(link_state.m_worldOrientation[0]);
            obs.push_back(link_state.m_worldOrientation[1]);
            obs.push_back(link_state.m_worldOrientation[2]);
            obs.push_back(link_state.m_worldOrientation[3]);

            obs.push_back(link_state.m_worldLinearVelocity[0]);
            obs.push_back(link_state.m_worldLinearVelocity[1]);
            obs.push_back(link_state.m_worldLinearVelocity[2]);

            obs.push_back(link_state.m_worldAngularVelocity[0]);
            obs.push_back(link_state.m_worldAngularVelocity[1]);
            obs.push_back(link_state.m_worldAngularVelocity[2]);
        }

        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    void render() override {
        // No-op
    }

    void modifySelectedJointPosition(double delta) {
        try {
            if (humanoid_ids.empty()) {
                printf("No humanoids available.\n");
                return;
            }
            if (selected_humanoid_id >= humanoid_ids.size()) {
                selected_humanoid_id = 0;
            }
            int humanoid_id = humanoid_ids[selected_humanoid_id];
            int num_joints = sim->getNumJoints(humanoid_id);
            if (selected_joint_index >= num_joints) {
                selected_joint_index = 0;
            }
            
            // Get current joint position
            b3JointSensorState state;
            if (sim->getJointState(humanoid_id, selected_joint_index, &state)) {
                double new_position = state.m_jointPosition + delta;
                
                // Apply the new position
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = new_position;
                motorArgs.m_targetVelocity = 1.0;
                motorArgs.m_maxTorqueValue = 100.0;
                sim->setJointMotorControl(humanoid_id, selected_joint_index, motorArgs);
                
                printf("Joint %d position: %.3f\n", selected_joint_index, new_position);
            }
        } catch (const std::exception& e) {
            printf("Exception in modifySelectedJointPosition: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in modifySelectedJointPosition.\n");
        }
    }

    void animate(int anim_skip_steps = 1) override {
        // Mouse interaction for joint manipulation
        static bool mouse_picking_active = false;
        static btVector3 last_mouse_pos;
        
        printf("Animation mode started. Press ESC to exit.\n");
        printf("Press 'S' to save joint positions, 'R' to reset to saved positions.\n");
        printf("Use 1/2 to select joint, 3/4 to modify position.\n");
        printf("Use 5/6 to select animation frame.\n");
        
        // Run animation loop until escape is pressed
        using clock = std::chrono::steady_clock;
        auto last_event_time = clock::now();
        while (true) {
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
                    if (humanoid_ids.empty()) {
                        printf("No humanoids available.\n");
                    } else {
                        int humanoid_id = humanoid_ids[selected_humanoid_id];
                        int num_joints = sim->getNumJoints(humanoid_id);
                        selected_joint_index = (selected_joint_index - 1 + num_joints) % num_joints;
                        
                        // Get and print the joint name
                        b3JointInfo jointInfo;
                        if (sim->getJointInfo(humanoid_id, selected_joint_index, &jointInfo)) {
                            printf("Selected joint: %d - %s\n", selected_joint_index, jointInfo.m_linkName);
                        } else {
                            printf("Selected joint: %d\n", selected_joint_index);
                        }
                    }
                }
                else if (event.m_keyCode == 50 && event.m_keyState == 1) { // '2' key - Next joint
                    if (humanoid_ids.empty()) {
                        printf("No humanoids available.\n");
                    } else {
                        int humanoid_id = humanoid_ids[selected_humanoid_id];
                        int num_joints = sim->getNumJoints(humanoid_id);
                        selected_joint_index = (selected_joint_index + 1) % num_joints;
                        
                        // Get and print the joint name
                        b3JointInfo jointInfo;
                        if (sim->getJointInfo(humanoid_id, selected_joint_index, &jointInfo)) {
                            printf("Selected joint: %d - %s\n", selected_joint_index, jointInfo.m_linkName);
                        } else {
                            printf("Selected joint: %d\n", selected_joint_index);
                        }
                    }
                }
                else if (event.m_keyCode == 51 && event.m_keyState == 1) { // '3' key - Increase position
                    modifySelectedJointPosition(0.1);
                    
                    // Print current joint value
                    if (!humanoid_ids.empty()) {
                        int humanoid_id = humanoid_ids[selected_humanoid_id];
                        b3JointSensorState state;
                        if (sim->getJointState(humanoid_id, selected_joint_index, &state)) {
                            printf("Joint %d position: %.3f\n", selected_joint_index, state.m_jointPosition);
                        }
                    }
                }
                else if (event.m_keyCode == 52 && event.m_keyState == 1) { // '4' key - Decrease position
                    modifySelectedJointPosition(-0.1);
                    
                    // Print current joint value
                    if (!humanoid_ids.empty()) {
                        int humanoid_id = humanoid_ids[selected_humanoid_id];
                        b3JointSensorState state;
                        if (sim->getJointState(humanoid_id, selected_joint_index, &state)) {
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
            for (int humanoid_id : humanoid_ids) {
                // Set the same angle as in reset function
                int i = 0; // Use first grid position for simplicity
                int j = 0;
                btVector3 start_pos(i * grid_space, j * grid_space, 1.0);
                btQuaternion start_ori;
                start_ori.setEulerZYX(0, M_PI_2, 0); // 90 degrees around Y-axis
                
                sim->resetBasePositionAndOrientation(humanoid_id, start_pos, start_ori);
                sim->resetBaseVelocity(humanoid_id, btVector3(0, 0, 0), btVector3(0, 0, 0));
            }
            
            // Step simulation to apply joint changes
            sim->stepSimulation();
            
            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }

    // External function to save joint positions
    void saveJointPositions() {
        try {
            if (humanoid_ids.empty()) {
                printf("No humanoids available to save.\n");
                return;
            }
            saved_joint_positions.clear();
            int humanoid_id = humanoid_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(humanoid_id);
            for (int j = 0; j < num_joints; ++j) {
                b3JointSensorState state;
                if (sim->getJointState(humanoid_id, j, &state)) {
                    saved_joint_positions.push_back(state.m_jointPosition);
                } else {
                    saved_joint_positions.push_back(0.0);
                }
            }
        } catch (const std::exception& e) {
            printf("Exception in saveJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in saveJointPositions.\n");
        }
    }

    // External function to reset joint positions
    void resetJointPositions() {
        try {
            if (humanoid_ids.empty()) {
                printf("No humanoids available to reset.\n");
                return;
            }
            if (saved_joint_positions.empty()) {
                printf("No saved positions to reset to.\n");
                return;
            }
            int humanoid_id = humanoid_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(humanoid_id);
            for (int j = 0; j < num_joints; ++j) {
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = saved_joint_positions[j];
                motorArgs.m_targetVelocity = 1.0; // Large value for instant movement
                motorArgs.m_maxTorqueValue = 100.0; // Large torque for instant movement
                sim->setJointMotorControl(humanoid_id, j, motorArgs);
            }
        } catch (const std::exception& e) {
            printf("Exception in resetJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in resetJointPositions.\n");
        }
    }
}; 