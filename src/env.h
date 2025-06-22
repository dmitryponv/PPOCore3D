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

    // Reset the environment and return initial observation
    virtual std::pair<torch::Tensor, std::unordered_map<std::string, float>> reset() = 0;

    // Step the environment with given action returns: observation, reward, terminated flag, truncated flag, extra info
    virtual std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> step(const torch::Tensor& action) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
};


using namespace std;

class AgentTargetEnv : public Env {
private:
    float x_min = -10.0f, x_max = 10.0f;
    float y_min = -10.0f, y_max = 10.0f;
    float max_force = 10.0f;

    int agent_id = -1;
    int target_id = -1;

    vector<float> agent_pos;  // {x, y}
    vector<float> target_pos; // {x, y}

    mt19937 rng;
    uniform_real_distribution<float> dist_x;
    uniform_real_distribution<float> dist_y;

public:
    AgentTargetEnv(torch::Device& device)
        : mDevice(device), dist_x(x_min, x_max), dist_y(y_min, y_max) {
        random_device rd;
        rng = mt19937(rd());

        sim = new b3RobotSimulatorClientAPI();
        bool isConnected = sim->connect(eCONNECT_GUI);

        if (!isConnected) {
            printf("Cannot connect\n");
            return;
        }

        sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
        sim->setTimeOut(10);
        sim->syncBodies();
        sim->setTimeStep(1. / 240.);
        sim->setGravity(btVector3(0, 0, -9.8));
        sim->loadURDF("plane.urdf");

        agent_id = sim->loadURDF("cube.urdf");
        target_id = sim->loadURDF("cube.urdf");

        //sim->changeDynamics(agent_id, -1, 1.0, 0.0, 0.0);
        //sim->changeDynamics(target_id, -1, 1.0, 0.0, 0.0);
    }

    Space observation_space() const override {
        return Space{ {4} };
    }

    Space action_space() const override {
        return Space{ {2} };
    }

    pair<torch::Tensor, unordered_map<string, float>> reset() override {
        agent_pos = { dist_x(rng), dist_y(rng) };
        target_pos = { dist_x(rng), dist_y(rng) };

        sim->resetBasePositionAndOrientation(agent_id, btVector3(agent_pos[0], agent_pos[1], 0.5f), btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        sim->resetBasePositionAndOrientation(target_id, btVector3(target_pos[0], target_pos[1], 0.5f), btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(target_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        return { get_observation(), {} };
    }

    tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) override {
        float dx = std::clamp(action[0].item<float>(), -1.0f, 1.0f);
        float dy = std::clamp(action[1].item<float>(), -1.0f, 1.0f);

        btVector3 agent_base_pos, target_base_pos;
        btQuaternion agent_q, target_q;

        sim->getBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);
        sim->getBasePositionAndOrientation(target_id, target_base_pos, target_q);

        agent_base_pos.setX(agent_base_pos.getX() + dx);
        agent_base_pos.setY(agent_base_pos.getY() + dy);

        sim->resetBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);

        agent_pos = { agent_base_pos.getX(), agent_base_pos.getY(), agent_base_pos.getZ() };
        target_pos = { target_base_pos.getX(), target_base_pos.getY(), target_base_pos.getZ() };

        float dist_x = agent_pos[0] - target_pos[0];
        float dist_y = agent_pos[1] - target_pos[1];
        float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

        float reward = -0.01f * distance;
        bool done = false;

        if (distance < 2.0f) {
            reward += 5.0f;
            done = true;
        }

        if (agent_pos[2] < 0.0f || target_pos[2] < 0.0f) {
            reward -= 5.0f;
            done = true;
        }

        return { get_observation(), reward, done, false, {} };
    }

    void render() override {
        printf("Agent: (%.2f, %.2f), Target: (%.2f, %.2f)\n", agent_pos[0], agent_pos[1], target_pos[0], target_pos[1]);
    }

private:
    torch::Device& mDevice;
    b3RobotSimulatorClientAPI* sim;

    torch::Tensor get_observation() const {
        return torch::tensor({ agent_pos[0], agent_pos[1], target_pos[0], target_pos[1] }).to(mDevice);
    }
};


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