#pragma once
#include "RobotSimulator.h"
#include "../env.h"
#include "../CommonInterfaces/CommonGUIHelperInterface.h"
#include "../CommonInterfaces/CommonExampleInterface.h"

class AgentTarget3dEnv : public Env3D {
private:
    float x_min = -10.0f, x_max = 10.0f;
    float y_min = -10.0f, y_max = 10.0f;
    float max_force = 10.0f;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist_x;
    std::uniform_real_distribution<float> dist_y;

public:
    AgentTarget3dEnv(torch::Device& device)
        : Env3D(device, new b3RobotSimulatorClientAPI()),
        x_min(-10.0f), x_max(10.0f),
        y_min(-10.0f), y_max(10.0f),
        max_force(10.0f),
        dist_x(x_min, x_max),
        dist_y(y_min, y_max)
    {
        std::random_device rd;
        rng = std::mt19937(rd());

        btVector3 base_pos(0.0f, 0.0f, 0.0f);

        b3RobotSimulatorLoadUrdfFileArgs plane_args;
        plane_args.m_startPosition = { base_pos.getX(), base_pos.getY(), base_pos.getZ() };
        plane_args.m_startOrientation = { 0, 0, 0, 1 };
        sim->loadURDF("plane.urdf", plane_args);

        b3RobotSimulatorLoadUrdfFileArgs obj_args;
        obj_args.m_startPosition = { base_pos.getX(), base_pos.getY(), base_pos.getZ() + 0.5f };
        obj_args.m_startOrientation = { 0, 0, 0, 1 };

        agent_id = sim->loadURDF("cube.urdf", obj_args);
        target_id = sim->loadURDF("cube.urdf", obj_args);
    }

    Space observation_space() const override {
        return Space{ {4} };
    }

    Space action_space() const override {
        return Space{ {2} };
    }

    torch::Tensor reset() override {
        btVector3 base_pos(0.0f, 0.0f, 0.0f);

        float ax = dist_x(rng), ay = dist_y(rng);
        float tx = dist_x(rng), ty = dist_y(rng);

        btVector3 agent_world_pos = base_pos + btVector3(ax, ay, 0.5f);
        btVector3 target_world_pos = base_pos + btVector3(tx, ty, 0.5f);

        sim->resetBasePositionAndOrientation(agent_id, agent_world_pos, btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        sim->resetBasePositionAndOrientation(target_id, target_world_pos, btQuaternion(0, 0, 0, 1));
        sim->resetBaseVelocity(target_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        
        float dx = std::clamp(actions[0].item<float>(), -1.0f, 1.0f);
        float dy = std::clamp(actions[1].item<float>(), -1.0f, 1.0f);

        btVector3 agent_base_pos, target_base_pos;
        btQuaternion agent_q, target_q;

        sim->getBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);
        sim->getBasePositionAndOrientation(target_id, target_base_pos, target_q);

        agent_base_pos.setX(agent_base_pos.getX() + dx);
        agent_base_pos.setY(agent_base_pos.getY() + dy);

        sim->resetBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);

        float dist_x = agent_base_pos.getX() - target_base_pos.getX();
        float dist_y = agent_base_pos.getY() - target_base_pos.getY();
        float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

        float reward = -0.01f * distance - 0.01f;
        bool done = false;

        //if (distance < 2.0f)
        //{
        //    reward += 5.0f;
        //    done = true;
        //}
        //if (agent_positions[i][2] < 0.0f || target_positions[i][2] < 0.0f) reward -= 5.0f, done = true;
        //if (dx == 0.0f && dy == 0.0f) reward -= 5.0f, done = true;
        //if (agent_positions[i][0] < x_min || agent_positions[i][0] > x_max ||
        //    agent_positions[i][1] < y_min || agent_positions[i][1] > y_max) reward -= 5.0f, done = true;

        return { get_observation(), reward, done, false };
    }

    void render() override {
        btVector3 agent_base_pos, target_base_pos;
        btQuaternion agent_q, target_q;
        
        sim->getBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);
        sim->getBasePositionAndOrientation(target_id, target_base_pos, target_q);
        
        printf("Agent: (%.2f, %.2f), Target: (%.2f, %.2f)\n",
            agent_base_pos.getX(), agent_base_pos.getY(),
            target_base_pos.getX(), target_base_pos.getY());
    }

    void animate() override {
        // Empty implementation
    }

private:
    torch::Tensor get_observation() const {
        btVector3 agent_base_pos, target_base_pos;
        btQuaternion agent_q, target_q;
        
        sim->getBasePositionAndOrientation(agent_id, agent_base_pos, agent_q);
        sim->getBasePositionAndOrientation(target_id, target_base_pos, target_q);
        
        return torch::tensor({
            agent_base_pos.getX(), agent_base_pos.getY(),
            target_base_pos.getX(), target_base_pos.getY()
            }).to(mDevice);
    }
};
