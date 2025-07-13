#pragma once
#include "RobotSimulator.h"
#include "env.h"

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
}; 