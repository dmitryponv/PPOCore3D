#pragma once
#include "../env.h"

class Humanoid3dEnv : public Env3D {
private:
    // std::vector<std::vector<b3LinkState>> saved_link_states; // Remove unused
    // std::vector<btVector3> saved_base_positions; // Remove unused
    // std::vector<btQuaternion> saved_base_orientations; // Remove unused

public:
    Humanoid3dEnv(torch::Device& device)
        : Env3D(device, new b3RobotSimulatorClientAPI()) // Pass sim pointer to base
    {
        start_ori.setEulerZYX(0, M_PI_2, 0); // 90 degrees around Y-axis
        start_pos = { 0,0,0.5 };

        btVector3 start_position(0.0f, 0.0f, 0.0f); // Use this->grid_space

        b3RobotSimulatorLoadUrdfFileArgs plane_args;
        plane_args.m_startPosition = { start_position.x(), start_position.y(), start_position.z() };
        plane_args.m_startOrientation = { 0.0f, 0.0f, 0.0f, 1.0f };
        sim->loadURDF("plane.urdf", plane_args);

        b3RobotSimulatorLoadUrdfFileArgs args;
        args.m_startPosition = { start_pos.x(), start_pos.y(), start_pos.z()};
        args.m_startOrientation = start_ori;
        args.m_useMultiBody = true;
        args.m_flags = 0;

        agent_id = sim->loadURDF("humanoid.urdf", args);

        sim->setRealTimeSimulation(false);
    }

    int observation_space() const override {
        int num_joints = sim->getNumJoints(agent_id);
        int obs_per_joint = 3 + 4 + 3 + 3; // pos + quat + linear vel + angular vel
        return num_joints * obs_per_joint;
    }

    int action_space() const override {
        return sim->getNumJoints(agent_id);
    }

    torch::Tensor reset() override {
        sim->resetBasePositionAndOrientation(agent_id, start_pos, start_ori);
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        int num_joints = sim->getNumJoints(agent_id);
        for (int k = 0; k < num_joints; ++k) {
            sim->resetJointState(agent_id, k, 0.0);
        }

        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;

        // Get animation for this frame
        auto anim = GetJointAnim(frame_index); //1 animation frame per second

        int num_joints = sim->getNumJoints(agent_id);
        const float max_velocity = 1.0f;
        // bool done = false; // done is determined after sim->stepSimulation()

        for (int j = 0; j < num_joints; ++j) {
            b3JointInfo jointInfo;
            sim->getJointInfo(agent_id, j, &jointInfo);

            if (jointInfo.m_jointType != JointType::eRevoluteType) {
                continue;
            }

            float action_single = actions[j].item<float>();
            float anim_val = (j < anim.size()) ? anim[j] : 0.0f;

            b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
            motorArgs.m_maxTorqueValue = 200.0f;
            motorArgs.m_targetPosition = action_single + anim_val;
            motorArgs.m_targetVelocity = max_velocity;

            sim->setJointMotorControl(agent_id, j, motorArgs);
        }

        sim->stepSimulation();

        // Find the link index for "torso_object"
        int torso_link_index = -1;
        for (int j = 0; j < num_joints; ++j) {
            b3JointInfo jointInfo;
            if (sim->getJointInfo(agent_id, j, &jointInfo)) {
                if (std::string(jointInfo.m_linkName) == "head_object") {
                    torso_link_index = j;
                    break;
                }
            }
        }

        btVector3 torso_pos(0, 0, 0);
        if (torso_link_index != -1) {
            b3LinkState torso_state;
            sim->getLinkState(agent_id, torso_link_index, 1, 0, &torso_state);
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

        GetFps();

        return { get_observation(), reward, done, false };
    }

    torch::Tensor get_observation() {
        std::vector<float> obs;
        int num_joints = sim->getNumJoints(agent_id);
        for (int j = 0; j < num_joints; ++j) {
            b3LinkState link_state;
            sim->getLinkState(agent_id, j, 1, 0, &link_state);

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