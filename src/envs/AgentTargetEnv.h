#pragma once
#include "RobotSimulator.h"
#include "env.h"
#include "../CommonInterfaces/CommonGUIHelperInterface.h"
#include "../CommonInterfaces/CommonExampleInterface.h"

class AgentTargetEnv : public Env3D {
private:
    float x_min = -10.0f, x_max = 10.0f;
    float y_min = -10.0f, y_max = 10.0f;
    float max_force = 10.0f;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist_x;
    std::uniform_real_distribution<float> dist_y;

public:
    AgentTargetEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f)
        : Env3D(device, new b3RobotSimulatorClientAPI()),
        x_min(-10.0f), x_max(10.0f),
        y_min(-10.0f), y_max(10.0f),
        max_force(10.0f),
        dist_x(x_min, x_max),
        dist_y(y_min, y_max)
    {
        this->grid_size = grid_size;
        this->grid_space = grid_space;

        std::random_device rd;
        rng = std::mt19937(rd());

        for (int i = 0; i < this->grid_size; ++i) {
            for (int j = 0; j < this->grid_size; ++j) {
                btVector3 base_pos(i * this->grid_space, j * this->grid_space, 0.0f);

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

        return get_observation(index);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions, int frame_index) override {
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

            float dist_x = agent_base_pos.getX() - target_base_pos.getX();
            float dist_y = agent_base_pos.getY() - target_base_pos.getY();
            float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

            float reward = -0.01f * distance - 0.01f;
            bool done = false;

            if (distance < 2.0f) reward += 5.0f, done = true;
            //if (agent_positions[i][2] < 0.0f || target_positions[i][2] < 0.0f) reward -= 5.0f, done = true;
            //if (dx == 0.0f && dy == 0.0f) reward -= 5.0f, done = true;
            //if (agent_positions[i][0] < x_min || agent_positions[i][0] > x_max ||
            //    agent_positions[i][1] < y_min || agent_positions[i][1] > y_max) reward -= 5.0f, done = true;

            results.emplace_back(get_observation(i), reward, done, false);
        }

        return results;
    }

    void render() override {
        for (size_t i = 0; i < agent_ids.size(); ++i) {
            btVector3 agent_base_pos, target_base_pos;
            btQuaternion agent_q, target_q;
            
            sim->getBasePositionAndOrientation(agent_ids[i], agent_base_pos, agent_q);
            sim->getBasePositionAndOrientation(target_ids[i], target_base_pos, target_q);
            
            printf("Agent %zu: (%.2f, %.2f), Target: (%.2f, %.2f)\n",
                i, agent_base_pos.getX(), agent_base_pos.getY(),
                target_base_pos.getX(), target_base_pos.getY());
        }
    }

    void animate() override {
        // Empty implementation
    }

private:
    torch::Tensor get_observation(int index) const {
        btVector3 agent_base_pos, target_base_pos;
        btQuaternion agent_q, target_q;
        
        sim->getBasePositionAndOrientation(agent_ids[index], agent_base_pos, agent_q);
        sim->getBasePositionAndOrientation(target_ids[index], target_base_pos, target_q);
        
        return torch::tensor({
            agent_base_pos.getX(), agent_base_pos.getY(),
            target_base_pos.getX(), target_base_pos.getY()
            }).to(mDevice);
    }
};
