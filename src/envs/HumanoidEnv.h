#pragma once
#include "env.h"

class HumanoidEnv : public Env3D {
private:
    // std::vector<std::vector<b3LinkState>> saved_link_states; // Remove unused
    // std::vector<btVector3> saved_base_positions; // Remove unused
    // std::vector<btQuaternion> saved_base_orientations; // Remove unused

public:
    HumanoidEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env3D(device, new b3RobotSimulatorClientAPI()) // Pass sim pointer to base
    {

        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

        start_ori.setEulerZYX(0, M_PI_2, 0); // 90 degrees around Y-axis
        start_pos = { 0,0,0.5 };
        agent_ids.clear();

        for (int i = 0; i < this->grid_size; ++i) { // Use this->grid_size
            for (int j = 0; j < this->grid_size; ++j) { // Use this->grid_size
                btVector3 start_position(i * this->grid_space, j * this->grid_space, 0.0f); // Use this->grid_space

                b3RobotSimulatorLoadUrdfFileArgs plane_args;
                plane_args.m_startPosition = { start_position.x(), start_position.y(), start_position.z() };
                plane_args.m_startOrientation = { 0.0f, 0.0f, 0.0f, 1.0f };
                sim->loadURDF("plane.urdf", plane_args);

                b3RobotSimulatorLoadUrdfFileArgs args;
                args.m_startPosition = { start_pos.x(), start_pos.y(), start_pos.z()};
                args.m_startOrientation = start_ori;
                args.m_useMultiBody = true;
                args.m_flags = 0;

                int id = sim->loadURDF("humanoid.urdf", args);
                agent_ids.push_back(id);
            }
        }

        sim->setRealTimeSimulation(false);
    }

    Space observation_space() const override {
        if (agent_ids.empty()) return Space{ {0} };
        int num_joints = sim->getNumJoints(agent_ids[0]);
        int obs_per_joint = 3 + 4 + 3 + 3; // pos + quat + linear vel + angular vel
        return Space{ {num_joints * obs_per_joint} };
    }

    Space action_space() const override {
        if (agent_ids.empty()) return Space{ {0} };
        return Space{ {sim->getNumJoints(agent_ids[0])} };
    }

    torch::Tensor reset(int index) override {
        if (index < 0 || index >= agent_ids.size()) {
            return torch::empty({ 0 });
        }

        int i = index / grid_size;
        int j = index % grid_size;

        int id = agent_ids[index];

        sim->resetBasePositionAndOrientation(id, start_pos, start_ori);
        sim->resetBaseVelocity(id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        int num_joints = sim->getNumJoints(id);
        for (int k = 0; k < num_joints; ++k) {
            sim->resetJointState(id, k, 0.0);
        }

        return get_observation(id);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions, int frame_index) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;

        static b3Clock clock;
        static double lastTime = clock.getTimeInSeconds();
        static int fpsTextId = -1;
        double currentTime = clock.getTimeInSeconds();
        static int frameCounterForFPS = 0;
        frameCounterForFPS++;
        double elapsed = currentTime - lastTime;
        if (elapsed >= 5.0) {
            double fps = frameCounterForFPS / elapsed;
            std::string text = "FPS: " + std::to_string(fps); // Fixed string concatenation
            if (fpsTextId >= 0)
                sim->removeUserDebugItem(fpsTextId);
            double pos[3] = { 0, 0, 2 };
            b3RobotSimulatorAddUserDebugTextArgs args;
            fpsTextId = sim->addUserDebugText(text.c_str(), pos, args);
            frameCounterForFPS = 0;
            lastTime = currentTime;
        }

        // Get animation for this frame
        auto anim = GetJointAnim(frame_index); //1 animation frame per second

        for (size_t i = 0; i < std::min(actions.size(), agent_ids.size()); ++i) {
            int id = agent_ids[i];
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

                float action_single = action[j].item<float>();
                float anim_val = (j < anim.size()) ? anim[j] : 0.0f;

                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_maxTorqueValue = 200.0f;
                motorArgs.m_targetPosition = action_single + anim_val;
                motorArgs.m_targetVelocity = max_velocity;

                sim->setJointMotorControl(id, j, motorArgs);
            }
        }

        sim->stepSimulation();

        for (size_t i = 0; i < agent_ids.size(); ++i) {
            int id = agent_ids[i];
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
            //float dist = (torso_pos - target_pos_check).length();
            //float reward = dist - 1.5f;
            float reward = (torso_pos[2] - target_pos_check[2]);
            bool done = false;// dist > 30;

            results.push_back({ get_observation(id), reward, done, false });

            //FOR TESTING ANIMATIONS
            //sim->resetBasePositionAndOrientation(id, start_pos, start_ori); sim->resetBaseVelocity(id, btVector3(0, 0, 0), btVector3(0, 0, 0));

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
}; 