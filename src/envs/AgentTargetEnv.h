#pragma once
#include "RobotSimulator.h"
#include "env.h"

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
