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
#include <unordered_map>

#include "RobotSimulator.h"

// Abstract environment interface
class Env {
public:
    struct Space {
        std::vector<int> shape;
    };

    virtual ~Env() = default;

    // Reset all environments and return a batch of observations
    virtual torch::Tensor reset(int index = -1) = 0;

    // Step all environments with a batch of actions
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>>> step(const std::vector<torch::Tensor>& actions) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
    int GetGridCount() { return (grid_size * grid_size); }

    int grid_size = 1;
    float grid_space = 20.0f;
};


using namespace std;

class AgentTargetEnv : public Env {
private:
    float x_min = -10.0f, x_max = 10.0f;
    float y_min = -10.0f, y_max = 10.0f;
    float max_force = 10.0f;

    vector<int> agent_ids;
    vector<int> target_ids;
    vector<vector<float>> agent_positions;
    vector<vector<float>> target_positions;

    mt19937 rng;
    uniform_real_distribution<float> dist_x;
    uniform_real_distribution<float> dist_y;

public:
    AgentTargetEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : mDevice(device), dist_x(x_min, x_max), dist_y(y_min, y_max) {
        this->grid_size = grid_size;
        this->grid_space = grid_space;

        std::random_device rd;
        rng = std::mt19937(rd());

        sim = new b3RobotSimulatorClientAPI();
        if (!sim->connect(eCONNECT_GUI)) {
            printf("Cannot connect\n");
            return;
        }

        sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
        sim->setTimeOut(10);
        sim->syncBodies();
        sim->setTimeStep(1. / 240.);
        sim->setGravity(btVector3(0, 0, -9.8));

        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                btVector3 base_pos(i * grid_space, j * grid_space, 0.0f);

                b3RobotSimulatorLoadUrdfFileArgs plane_args;
                plane_args.m_startPosition = { base_pos.getX(), base_pos.getY(), base_pos.getZ() };
                plane_args.m_startOrientation = { 0, 0, 0, 1 };
                sim->loadURDF("plane.urdf", plane_args);

                b3RobotSimulatorLoadUrdfFileArgs obj_args;
                obj_args.m_startPosition = { base_pos.getX(), base_pos.getY(), base_pos.getZ() + 0.5f };
                obj_args.m_startOrientation = { 0, 0, 0, 1 };

                int agent_id = sim->loadURDF("cube.urdf", obj_args);
                int target_id = sim->loadURDF("cube.urdf", obj_args);

                agent_ids.push_back(agent_id);
                target_ids.push_back(target_id);
                agent_positions.emplace_back(3);
                target_positions.emplace_back(3);
            }
        }
    }

    Space observation_space() const override {
        return Space{ {4} };
    }

    Space action_space() const override {
        return Space{ {2} };
    }

    torch::Tensor reset(int index = -1) override {
        if (index < 0 || index >= agent_ids.size()) return torch::zeros({ 4 });

        int i = index / grid_size;
        int j = index % grid_size;

        btVector3 base_pos(i * grid_space, j * grid_space, 0.0f);

        float ax = dist_x(rng), ay = dist_y(rng);
        float tx = dist_x(rng), ty = dist_y(rng);

        btVector3 agent_world_pos = base_pos + btVector3(ax, ay, 0.5f);
        btVector3 target_world_pos = base_pos + btVector3(tx, ty, 0.5f);

        sim->resetBasePositionAndOrientation(agent_ids[index], agent_world_pos, btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(agent_ids[index], btVector3(0, 0, 0), btVector3(0, 0, 0));

        sim->resetBasePositionAndOrientation(target_ids[index], target_world_pos, btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(target_ids[index], btVector3(0, 0, 0), btVector3(0, 0, 0));

        agent_positions[index] = { agent_world_pos.getX(), agent_world_pos.getY(), agent_world_pos.getZ() };
        target_positions[index] = { target_world_pos.getX(), target_world_pos.getY(), target_world_pos.getZ() };

        return get_observation(index);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>>>
        step(const std::vector<torch::Tensor>& actions) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>>> results;

        for (size_t i = 0; i < actions.size(); ++i) {
            float dx = std::clamp(actions[i][0].item<float>(), -1.0f, 1.0f);
            float dy = std::clamp(actions[i][1].item<float>(), -1.0f, 1.0f);

            btVector3 agent_base_pos, target_base_pos;
            btQuaternion agent_q, target_q;

            sim->getBasePositionAndOrientation(agent_ids[i], agent_base_pos, agent_q);
            sim->getBasePositionAndOrientation(target_ids[i], target_base_pos, target_q);

            agent_base_pos.setX(agent_base_pos.getX() + dx);
            agent_base_pos.setY(agent_base_pos.getY() + dy);

            sim->resetBasePositionAndOrientation(agent_ids[i], agent_base_pos, agent_q);

            agent_positions[i] = { agent_base_pos.getX(), agent_base_pos.getY(), agent_base_pos.getZ() };
            target_positions[i] = { target_base_pos.getX(), target_base_pos.getY(), target_base_pos.getZ() };

            float dist_x = agent_positions[i][0] - target_positions[i][0];
            float dist_y = agent_positions[i][1] - target_positions[i][1];
            float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

            float reward = -0.01f * distance - 0.01f;
            bool done = false;

            if (distance < 2.0f) reward += 5.0f, done = true;
            if (agent_positions[i][2] < 0.0f || target_positions[i][2] < 0.0f) reward -= 5.0f, done = true;
            if (dx == 0.0f && dy == 0.0f) reward -= 5.0f, done = true;
            //if (agent_positions[i][0] < x_min || agent_positions[i][0] > x_max ||
            //    agent_positions[i][1] < y_min || agent_positions[i][1] > y_max) reward -= 5.0f, done = true;

            results.emplace_back(get_observation(i), reward, done, false, std::unordered_map<std::string, float>{});
        }

        return results;
    }

    void render() override {
        for (size_t i = 0; i < agent_positions.size(); ++i) {
            printf("Agent %zu: (%.2f, %.2f), Target: (%.2f, %.2f)\n",
                i, agent_positions[i][0], agent_positions[i][1],
                target_positions[i][0], target_positions[i][1]);
        }
    }

private:
    torch::Device& mDevice;
    b3RobotSimulatorClientAPI* sim;

    torch::Tensor get_observation(int index) const {
        return torch::tensor({
            agent_positions[index][0], agent_positions[index][1],
            target_positions[index][0], target_positions[index][1]
            }).to(mDevice);
    }
};


/*
class PendulumEnv : public Env {
public:
    PendulumEnv(torch::Device& device, const string& render_mode = "", float gravity = 10.0f)
        : mDevice(device), g(gravity), render_mode(render_mode) {
        max_speed = 8.0f;
        max_torque = 2.0f;
        dt = 0.05f;
        m = 1.0f;
        l = 1.0f;
        last_u = 0.0f;

        obs_space.shape = { 3 };
        act_space.shape = { 1 };

        random_device rd;
        rng = mt19937(rd());
        dist = uniform_real_distribution<float>(-3.14f, 3.14f);
    }

    pair<torch::Tensor, unordered_map<string, float>> reset() override {
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        state = { theta, theta_dot };
        last_u = 0.0f;
        return { get_obs(), {} };
    }

    tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) override {
        float u = std::clamp(action.item<float>(), -max_torque, max_torque);
        last_u = u;

        float theta = state[0];
        float theta_dot = state[1];

        float cost = angle_normalize(theta) * angle_normalize(theta)
            + 0.1f * theta_dot * theta_dot
            + 0.001f * u * u;

        float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * std::sin(theta) + 3.0f / (m * l * l) * u) * dt;
        new_theta_dot = std::clamp(new_theta_dot, -max_speed, max_speed);
        float new_theta = theta + new_theta_dot * dt;

        state = { new_theta, new_theta_dot };

        return { get_obs(), -cost, false, false, {} };
    }

    void render() override {
        if (render_mode == "human") {
            cout << "Angle: " << state[0] << ", Angular velocity: " << state[1] << ", Torque: " << last_u << "\n";
        }
    }

    Space observation_space() const override {
        return obs_space;
    }

    Space action_space() const override {
        return act_space;
    }

private:
    torch::Device& mDevice;
    float g, m, l, dt;
    float max_speed, max_torque;
    float last_u;
    string render_mode;
    vector<float> state;
    Space obs_space, act_space;

    mt19937 rng;
    uniform_real_distribution<float> dist;

    torch::Tensor get_obs() const {
        float theta = state[0];
        float theta_dot = state[1];
        return torch::tensor({ std::cos(theta), std::sin(theta), theta_dot }).to(mDevice);
    }

    float angle_normalize(float x) const {
        return fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }

    float clamp(float v, float lo, float hi) const {
        return std::max(lo, std::min(v, hi));
    }
};

class RobotEnv : public Env {
public:
    RobotEnv(torch::Device& device);

    Space observation_space() const override;

    Space action_space() const override;

    std::pair<torch::Tensor, std::unordered_map<std::string, float>> reset() override;

    std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> step(const torch::Tensor& action) override;

    void render() override;

private:
    torch::Device& mDevice;
    b3RobotSimulatorClientAPI* sim;
    int minitaurUid = -1;
    int numJoints = 0;
    std::vector<int> validTorqueJoints;
    torch::Tensor get_observation() const;
};
*/


class HumanoidEnv : public Env {
private:
    b3RobotSimulatorClientAPI* sim;
    std::vector<int> humanoid_ids;
    torch::Device& mDevice;
public:
    HumanoidEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : mDevice(device) {

        this->grid_size = grid_size;
        this->grid_space = grid_space;
        sim = new b3RobotSimulatorClientAPI();
        if (!sim->connect(eCONNECT_GUI)) {
            printf("Cannot connect\n");
            return;
        }

        sim->setGravity(btVector3(0, 0, -9.8));
        sim->setTimeStep(1. / 240.);

        humanoid_ids.clear();

        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                btVector3 start_pos(i * grid_space, j * grid_space, 0.0f);

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

    vector<tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>>> step(const vector<torch::Tensor>& actions) {
        vector<tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>>> results;

        static b3Clock clock;
        static double lastTime = clock.getTimeInSeconds();
        static int frameCount = 0;
        static int fpsTextId = -1;
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

        for (size_t i = 0; i < std::min(actions.size(), humanoid_ids.size()); ++i) {
            int id = humanoid_ids[i];
            const torch::Tensor& action = actions[i];
            int num_joints = sim->getNumJoints(id);
            const float max_velocity = 1.0f;
            bool done = false;

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
            float reward = dist - 1.0f;
            bool done = dist > 30;

            results.push_back({ get_observation(id), reward, done, false, {} });
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
