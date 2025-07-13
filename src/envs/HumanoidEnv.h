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

            b3LinkState head_state;
            sim->getLinkState(id, num_joints - 1, 0, 0, &head_state);

            btVector3 head_pos(
                head_state.m_worldPosition[0],
                head_state.m_worldPosition[1],
                head_state.m_worldPosition[2]
            );

            btVector3 target_pos_check(0.0f, 0.0f, 2.0f);
            float dist = (head_pos - target_pos_check).length();
            float reward = dist - 2.0f;
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

    // Raycast function similar to the provided example
    int raycast(float mouseX, float mouseY) {
        // Convert mouse coordinates to world ray
        // For simplicity, we'll create a ray from the camera position
        btVector3 rayFrom(mouseX, mouseY, 10.0f);
        btVector3 rayTo(mouseX, mouseY, -10.0f);
        
        // Use the collision world for raycasting
        btCollisionWorld::ClosestRayResultCallback rayCallback(rayFrom, rayTo);
        // Note: We need access to collisionWorld_ from the simulator
        // For now, we'll use a simplified approach
        
        // Check if any humanoid is near the ray
        for (int humanoid_id : humanoid_ids) {
            btVector3 base_pos;
            btQuaternion base_ori;
            sim->getBasePositionAndOrientation(humanoid_id, base_pos, base_ori);
            
            // Simple distance check - if mouse is close to humanoid base position
            float dist_x = mouseX - base_pos.getX();
            float dist_y = mouseY - base_pos.getY();
            float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);
            
            if (distance < 5.0f) { // Within 5 units
                return humanoid_id;
            }
        }
        
        return -1; // No hit
    }

    void animate(int anim_skip_steps = 1) override {
        // Mouse interaction for joint manipulation
        static bool mouse_picking_active = false;
        static int selected_humanoid_id = -1;
        static int selected_joint_id = -1;
        static btVector3 last_mouse_pos;
        static std::vector<float> fixed_joint_angles; // Store fixed joint angles
        
        printf("Animation mode started. Press ESC to exit.\n");
        
        // Run animation loop until escape is pressed
        while (true) {
            // Check for escape key
            b3KeyboardEventsData keyboardEvents;
            sim->getKeyboardEvents(&keyboardEvents);
            
            for (int i = 0; i < keyboardEvents.m_numKeyboardEvents; i++) {
                const b3KeyboardEvent& event = keyboardEvents.m_keyboardEvents[i];
                if (event.m_keyCode == 27) { // ESC key
                    printf("Animation mode exited.\n");
                    return;
                }
            }
            
            // Get mouse state from the simulator
            b3MouseEventsData mouseEvents;
            sim->getMouseEvents(&mouseEvents);
            
            // Handle mouse events for joint picking and manipulation
            for (int i = 0; i < mouseEvents.m_numMouseEvents; i++) {
                const b3MouseEvent& event = mouseEvents.m_mouseEvents[i];
                
                if (event.m_eventType == 1) { // MOUSE_BUTTON_CLICKED
                    if (event.m_buttonIndex == 0) { // Left mouse button
                        // Perform raycast to pick joint
                        btVector3 rayFromWorld(event.m_mousePosX, event.m_mousePosY, 10.0f);
                        btVector3 rayToWorld(event.m_mousePosX, event.m_mousePosY, -10.0f);
                        
                        // Use raycast instead of pick body
                        int hit_humanoid = raycast(event.m_mousePosX, event.m_mousePosY);
                        
                        if (hit_humanoid >= 0) {
                            selected_humanoid_id = hit_humanoid;
                            
                            // For simplicity, select the first joint
                            selected_joint_id = 0;
                            
                            // Store current joint angles to fix them in place
                            fixed_joint_angles.clear();
                            int num_joints = sim->getNumJoints(selected_humanoid_id);
                            for (int j = 0; j < num_joints; j++) {
                                b3LinkState link_state;
                                sim->getLinkState(selected_humanoid_id, j, 1, 0, &link_state);
                                float angle = 0.0f;
                                if (link_state.m_worldOrientation[0] != 0.0f || link_state.m_worldOrientation[1] != 0.0f || 
                                    link_state.m_worldOrientation[2] != 0.0f || link_state.m_worldOrientation[3] != 0.0f) {
                                    float qw = link_state.m_worldOrientation[3];
                                    float qz = link_state.m_worldOrientation[2];
                                    angle = 2.0f * std::atan2(qz, qw);
                                }
                                fixed_joint_angles.push_back(angle);
                            }
                            
                            mouse_picking_active = true;
                            last_mouse_pos = btVector3(event.m_mousePosX, event.m_mousePosY, 0);
                            printf("Selected joint %d on humanoid %d\n", selected_joint_id, selected_humanoid_id);
                        }
                    }
                }
                else if (event.m_eventType == 2) { // MOUSE_BUTTON_RELEASED
                    if (event.m_buttonIndex == 0) { // Left mouse button released
                        mouse_picking_active = false;
                        selected_humanoid_id = -1;
                        selected_joint_id = -1;
                        fixed_joint_angles.clear();
                        printf("Released joint selection\n");
                    }
                }
                else if (event.m_eventType == 3 && mouse_picking_active) { // MOUSE_MOVED
                    // Handle joint manipulation when mouse is dragged
                    if (selected_humanoid_id >= 0 && selected_joint_id >= 0) {
                        btVector3 current_mouse_pos(event.m_mousePosX, event.m_mousePosY, 0);
                        btVector3 mouse_delta = current_mouse_pos - last_mouse_pos;
                        
                        // Get current joint state
                        b3JointInfo jointInfo;
                        sim->getJointInfo(selected_humanoid_id, selected_joint_id, &jointInfo);
                        
                        if (jointInfo.m_jointType == JointType::eRevoluteType) {
                            // Calculate joint angle change based on mouse movement
                            float angle_change = mouse_delta.getX() * 0.01f; // Sensitivity
                            
                            // Get current joint angle using link state
                            b3LinkState link_state;
                            sim->getLinkState(selected_humanoid_id, selected_joint_id, 1, 0, &link_state);
                            // For a revolute joint, estimate the angle from orientation (assuming rotation about Z)
                            float current_angle = 0.0f;
                            if (link_state.m_worldOrientation[0] != 0.0f || link_state.m_worldOrientation[1] != 0.0f || link_state.m_worldOrientation[2] != 0.0f || link_state.m_worldOrientation[3] != 0.0f) {
                                // Convert quaternion to angle (assuming Z axis)
                                float qw = link_state.m_worldOrientation[3];
                                float qz = link_state.m_worldOrientation[2];
                                current_angle = 2.0f * std::atan2(qz, qw); // Only valid for Z axis rotation
                            }
                            float new_angle = current_angle + angle_change;
                            
                            // Apply joint control to all joints - fix others in place
                            int num_joints = sim->getNumJoints(selected_humanoid_id);
                            for (int j = 0; j < num_joints; j++) {
                                b3JointInfo joint_info;
                                sim->getJointInfo(selected_humanoid_id, j, &joint_info);
                                
                                if (joint_info.m_jointType == JointType::eRevoluteType) {
                                    b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                                    motorArgs.m_maxTorqueValue = 200.0f;
                                    
                                    if (j == selected_joint_id) {
                                        // Animate the selected joint
                                        motorArgs.m_targetPosition = new_angle;
                                        motorArgs.m_targetVelocity = 1.0f;
                                    } else {
                                        // Fix other joints in their current positions
                                        motorArgs.m_targetPosition = fixed_joint_angles[j];
                                        motorArgs.m_targetVelocity = 0.0f; // No movement
                                    }
                                    
                                    sim->setJointMotorControl(selected_humanoid_id, j, motorArgs);
                                }
                            }
                            
                            printf("Joint %d angle: %.3f\n", selected_joint_id, new_angle);
                        }
                        
                        last_mouse_pos = current_mouse_pos;
                    }
                }
            }
            
            // Step simulation to apply joint changes
            sim->stepSimulation();
            
            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }
}; 