#pragma once
#include "RobotSimulator.h"
#include "env.h"

class RobotEnv : public Env3D {
public:
    // Modified constructor to take grid_size and grid_space
    RobotEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env3D(device, new b3RobotSimulatorClientAPI())
    {
        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

        int total_robots = GetGridCount(); // Use GetGridCount()
        object_ids.reserve(total_robots);

        bool valid_joints_initialized = false; // Flag to initialize validTorqueJoints once

        for (int i = 0; i < this->grid_size; ++i) { // Use this->grid_size
            for (int j = 0; j < this->grid_size; ++j) { // Use this->grid_size
                btVector3 base_pos(i * this->grid_space, j * this->grid_space, 0.0f); // Use this->grid_space

                // Load plane for each grid cell
                b3RobotSimulatorLoadUrdfFileArgs plane_args;
                plane_args.m_startPosition = { base_pos.getX(), base_pos.getY(), base_pos.getZ() };
                plane_args.m_startOrientation = { 0, 0, 0, 1 };
                sim->loadURDF("plane.urdf", plane_args);

                MinitaurSetup minitaur_setup; // Create a new setup object for each robot if it manages unique properties
                int current_minitaur_uid = minitaur_setup.setupMinitaur(sim, base_pos + btVector3(0, 0, .3));
                object_ids.push_back(current_minitaur_uid);

                if (!valid_joints_initialized) { // Only analyze joints for the first robot
                    int current_num_joints = sim->getNumJoints(current_minitaur_uid);
                    for (int k = 0; k < current_num_joints; ++k) {
                        b3JointInfo jointInfo;
                        sim->getJointInfo(current_minitaur_uid, k, &jointInfo);
                        std::string jointName(jointInfo.m_linkName);
                        btVector3 axis(jointInfo.m_jointAxis[0], jointInfo.m_jointAxis[1], jointInfo.m_jointAxis[2]);

                        // Modified condition: only check for "motor"
                        if (!axis.fuzzyZero() && jointName.find("motor") != std::string::npos) {
                            validTorqueJoints.push_back(k);
                        }
                    }
                    valid_joints_initialized = true; // Mark as initialized
                }
            }
        }

        sim->setRealTimeSimulation(false);
    }

    Space observation_space() const override {
        // Observations for all joints (positions + velocities) + base pos/vel
        // Assuming all minitaurs have the same number of valid torque joints for the action space.
        // For observation, we take the first minitaur's info if it exists.
        if (object_ids.empty()) return Space{ {0} };
        int num_joints_first_robot = sim->getNumJoints(object_ids[0]);
        return Space{ {6 + num_joints_first_robot * 6} };
    }

    Space action_space() const override {
        // Actions only for joints with non-zero axis
        if (validTorqueJoints.empty()) return Space{ {0} }; // Use the single validTorqueJoints
        return Space{ {static_cast<int>(validTorqueJoints.size())} }; // Use the single validTorqueJoints
    }

    torch::Tensor reset(int index = -1) override {

        // Reset a specific environment
        int current_uid = object_ids[index];
        int curr_i = index / grid_size;
        int curr_j = index % grid_size;
        btVector3 base_pos_offset(curr_i * grid_space, curr_j * grid_space, 0.0f);

        sim->resetBasePositionAndOrientation(current_uid, base_pos_offset + btVector3(0, 0, 0.3), btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(current_uid, btVector3(0, 0, 0), btVector3(0, 0, 0));

        int num_joints = sim->getNumJoints(current_uid);
        for (int j = 0; j < num_joints; ++j) {
            sim->resetJointState(current_uid, j, 0.0);
        }

        return get_observation(current_uid);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;
        results.reserve(actions.size());

        static b3Clock clock;
        static double lastTime = clock.getTimeInSeconds();
        static int frameCount = 0;
        static int fpsTextId = -1;

        for (size_t i = 0; i < std::min(actions.size(), object_ids.size()); ++i) {
            int current_uid = object_ids[i];
            const torch::Tensor& action = actions[i];
            const std::vector<int>& current_valid_torque_joints = validTorqueJoints; // Use the single validTorqueJoints

            for (size_t k = 0; k < current_valid_torque_joints.size(); ++k) {
                int joint_idx = current_valid_torque_joints[k];
                float torque = action[k].item<float>() * 10.0f;
                torque = std::clamp(torque, -5.0f, 5.0f);
                b3JointInfo jointInfo;
                sim->getJointInfo(current_uid, joint_idx, &jointInfo);
                btVector3 axis(jointInfo.m_jointAxis[0], jointInfo.m_jointAxis[1], jointInfo.m_jointAxis[2]);
                btVector3 torqueVec = axis.normalized() * torque;

                sim->applyExternalTorque(current_uid, joint_idx, torqueVec, EF_LINK_FRAME);
            }
        }

        sim->stepSimulation(); // Step simulation once for all robots

        frameCount++;
        double currentTime = clock.getTimeInSeconds();
        double elapsed = currentTime - lastTime;

        if (elapsed >= 1.0) {
            double fps = frameCount / elapsed;
            std::string text = "FPS: " + std::to_string(fps);
            if (fpsTextId >= 0)
                sim->removeUserDebugItem(fpsTextId);
            double pos[3] = { 0, 0, 2 };
            b3RobotSimulatorAddUserDebugTextArgs args;
            fpsTextId = sim->addUserDebugText(text.c_str(), pos, args);
            lastTime = currentTime;
            frameCount = 0;
        }

        // Collect results for all robots after simulation step
        for (size_t i = 0; i < object_ids.size(); ++i) {
            int current_uid = object_ids[i];
            b3LinkState baseLinkState;
            sim->getLinkState(current_uid, 0, 1, 1, &baseLinkState);
            float x = baseLinkState.m_worldPosition[0];
            float y = baseLinkState.m_worldPosition[1];
            float z = baseLinkState.m_worldPosition[2];

            float reward = z - 0.05f - 0.1f * std::sqrt(x * x + y * y);
            bool done = false; // Add specific done condition if needed, e.g., if z < 0.1

            results.push_back({ get_observation(current_uid), reward, done, false });
        }
        return results;
    }

    void render() override {
        // No-op specific rendering for RobotEnv in this setup, as GUI connection handles visual
    }

private:
    std::vector<int> validTorqueJoints; // Single vector for valid torque joints, assumed same for all robots

    torch::Tensor get_observation(int uid) const {
        std::vector<float> obs;

        // Base link state (link 0) for the given robot UID
        b3LinkState baseLinkState;
        sim->getLinkState(uid, 0, 1, 1, &baseLinkState);
        obs.push_back(baseLinkState.m_worldPosition[0]);
        obs.push_back(baseLinkState.m_worldPosition[1]);
        obs.push_back(baseLinkState.m_worldPosition[2]);
        obs.push_back(baseLinkState.m_worldLinearVelocity[0]);
        obs.push_back(baseLinkState.m_worldLinearVelocity[1]);
        obs.push_back(baseLinkState.m_worldLinearVelocity[2]);

        // All joint link states for the given robot UID
        int numJoints = sim->getNumJoints(uid);
        for (int j = 0; j < numJoints; ++j) {
            b3LinkState linkState;
            sim->getLinkState(uid, j, 1, 1, &linkState);

            obs.push_back(linkState.m_worldPosition[0]);
            obs.push_back(linkState.m_worldPosition[1]);
            obs.push_back(linkState.m_worldPosition[2]);
            obs.push_back(linkState.m_worldLinearVelocity[0]);
            obs.push_back(linkState.m_worldLinearVelocity[1]);
            obs.push_back(linkState.m_worldLinearVelocity[2]);
        }

        return torch::from_blob(obs.data(), { (int64_t)obs.size() }).clone().to(mDevice);
    }
}; 