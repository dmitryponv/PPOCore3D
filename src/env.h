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
#include <algorithm> // Required for std::clamp, std::min, std::max

#include "RobotSimulator.h" // Assuming this includes MinitaurSetup.h if needed by RobotEnv

// Abstract environment interface
class Env {
public:
    struct Space {
        std::vector<int> shape;
    };

    virtual ~Env() = default;

    // Reset a specific environment and return its observation
    virtual torch::Tensor reset(int index = -1) = 0;

    // Step all environments with a batch of actions
    // Removed std::unordered_map from the tuple
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
    int GetGridCount() const { return (grid_size * grid_size); } // Made const

    int grid_size = 1;
    float grid_space = 20.0f;
};

class AgentTargetEnv : public Env {
private:
    float x_min = -10.0f, x_max = 10.0f;
    float y_min = -10.0f, y_max = 10.0f;
    float max_force = 10.0f;

    std::vector<int> agent_ids;
    std::vector<int> target_ids;
    std::vector<std::vector<float>> agent_positions;
    std::vector<std::vector<float>> target_positions;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist_x;
    std::uniform_real_distribution<float> dist_y;

public:
    // Constructor now correctly initializes base class members
    AgentTargetEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env(), // Call base class constructor
        mDevice(device),
        x_min(-10.0f), x_max(10.0f),
        y_min(-10.0f), y_max(10.0f),
        max_force(10.0f),
        dist_x(x_min, x_max),
        dist_y(y_min, y_max)
    {
        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

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

        for (int i = 0; i < this->grid_size; ++i) { // Use this->grid_size
            for (int j = 0; j < this->grid_size; ++j) { // Use this->grid_size
                btVector3 base_pos(i * this->grid_space, j * this->grid_space, 0.0f); // Use this->grid_space

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
        // Only handles individual environment resets
        if (index < 0 || index >= agent_ids.size()) {
            // Return an empty tensor or throw an error for invalid index, as a reset is expected for a specific environment.
            // Returning an empty tensor as per typical Torch C++ API usage for empty results.
            return torch::empty({ 0 });
        }

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

    std::vector<std::tuple<torch::Tensor, float, bool, bool>>
        step(const std::vector<torch::Tensor>& actions) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;
        results.reserve(actions.size());

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

            results.emplace_back(get_observation(i), reward, done, false);
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


class PendulumEnv : public Env {
public:
    // Modified constructor to take grid_size and grid_space
    PendulumEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f, const std::string& render_mode = "", float gravity = 10.0f)
        : Env(), // Call base class constructor
        mDevice(device),
        g(gravity),
        render_mode(render_mode)
    {
        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

        max_speed = 8.0f;
        max_torque = 2.0f;
        dt = 0.05f;
        m = 1.0f;
        l = 1.0f;

        obs_space.shape = { 3 };
        act_space.shape = { 1 };

        std::random_device rd;
        rng = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-3.14f, 3.14f);

        states.resize(GetGridCount(), std::vector<float>(2)); // Use GetGridCount()
        last_us.resize(GetGridCount(), 0.0f); // Use GetGridCount()
    }

    torch::Tensor reset(int index = -1) override {
        // Only handles individual environment resets
        if (index < 0 || index >= states.size()) {
            return torch::empty({ 0 }); // Invalid index, return empty tensor
        }

        // Reset a specific environment
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        states[index] = { theta, theta_dot };
        last_us[index] = 0.0f;
        return get_obs(index);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;
        results.reserve(actions.size());

        for (size_t i = 0; i < actions.size(); ++i) {
            float u = std::clamp(actions[i].item<float>(), -max_torque, max_torque);
            last_us[i] = u;

            float theta = states[i][0];
            float theta_dot = states[i][1];

            float cost = angle_normalize(theta) * angle_normalize(theta)
                + 0.1f * theta_dot * theta_dot
                + 0.001f * u * u;

            float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * std::sin(theta) + 3.0f / (m * l * l) * u) * dt;
            new_theta_dot = std::clamp(new_theta_dot, -max_speed, max_speed);
            float new_theta = theta + new_theta_dot * dt;

            states[i] = { new_theta, new_theta_dot };

            results.emplace_back(get_obs(i), -cost, false, false);
        }
        return results;
    }

    void render() override {
        if (render_mode == "human") {
            // Render only the first pendulum for simplicity, or modify to loop and print all.
            // For general rendering of multiple environments, this method would need a major overhaul
            // to display multiple simulation windows or a unified view.
            if (!states.empty()) {
                std::cout << "Angle: " << states[0][0] << ", Angular velocity: " << states[0][1] << ", Torque: " << last_us[0] << "\n";
            }
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
    std::string render_mode;
    std::vector<std::vector<float>> states; // State for multiple pendulums
    std::vector<float> last_us; // Last applied torque for multiple pendulums
    Space obs_space, act_space;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    torch::Tensor get_obs(int index) const {
        float theta = states[index][0];
        float theta_dot = states[index][1];
        return torch::tensor({ std::cos(theta), std::sin(theta), theta_dot }).to(mDevice);
    }

    float angle_normalize(float x) const {
        return std::fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }
};

class RobotEnv : public Env {
public:
    // Modified constructor to take grid_size and grid_space
    RobotEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env(), // Call base class constructor
        mDevice(device)
    {
        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

        sim = new b3RobotSimulatorClientAPI();
        bool isConnected = sim->connect(eCONNECT_GUI);

        if (!isConnected) {
            printf("Cannot connect\n");
            return;
        }

        sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
        sim->setTimeOut(10);
        sim->syncBodies();

        btScalar fixedTimeStep = 1. / 240.;
        sim->setTimeStep(fixedTimeStep);
        sim->setGravity(btVector3(0, 0, -9.8));

        int total_robots = GetGridCount(); // Use GetGridCount()
        minitaurUids.reserve(total_robots);
        // validTorqueJoints_per_robot removed

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
                minitaurUids.push_back(current_minitaur_uid);

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
        if (minitaurUids.empty()) return Space{ {0} };
        int num_joints_first_robot = sim->getNumJoints(minitaurUids[0]);
        return Space{ {6 + num_joints_first_robot * 6} };
    }

    Space action_space() const override {
        // Actions only for joints with non-zero axis
        if (validTorqueJoints.empty()) return Space{ {0} }; // Use the single validTorqueJoints
        return Space{ {static_cast<int>(validTorqueJoints.size())} }; // Use the single validTorqueJoints
    }

    torch::Tensor reset(int index = -1) override {

        // Reset a specific environment
        int current_uid = minitaurUids[index];
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

        for (size_t i = 0; i < std::min(actions.size(), minitaurUids.size()); ++i) {
            int current_uid = minitaurUids[i];
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
        for (size_t i = 0; i < minitaurUids.size(); ++i) {
            int current_uid = minitaurUids[i];
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
    torch::Device& mDevice;
    b3RobotSimulatorClientAPI* sim;
    std::vector<int> minitaurUids; // UIDs for all minitaur robots
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
            float reward = dist - 1.0f;
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
