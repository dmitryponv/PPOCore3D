#pragma once
#include "RobotSimulator.h"
#include "../env.h"
#include <vector>
#include <stdexcept>
#include <string>

// A 3D environment for simulating a basketball agent controlled by forces.
class Basketball3dEnv : public Env3D {
public:
    // Constructor. Loads the URDF and sets up the environment.
    Basketball3dEnv(torch::Device& device)
        : Env3D(device, new b3RobotSimulatorClientAPI())
    {
        // Load a plane for the ground.
        b3RobotSimulatorLoadUrdfFileArgs plane_args;
        plane_args.m_startPosition = { 0.0f, 0.0f, 0.0f };
        sim->loadURDF("plane.urdf", plane_args);

        // Load a sphere as the agent. We'll use a URDF for simplicity.
        b3RobotSimulatorLoadUrdfFileArgs sphere_args;
        sphere_args.m_startPosition = { 0.0f, 0.0f, 2.0f }; // Start at target position
        sphere_args.m_useMultiBody = true;
        sphere_args.m_flags = 0;
        agent_id = sim->loadURDF("sphere2.urdf", sphere_args);

        // Set simulation parameters.
        sim->setGravity({ 0, 0, 0 });
        sim->setTimeStep(1.0 / 240.0);
        sim->setRealTimeSimulation(false);
    }

    // Resets the agent to its initial position and velocity.
    torch::Tensor reset() override {
        // Reset base state to the starting position and zero velocity.
        btVector3 start_pos(0.0f, 0.0f, 1.0f);
        btQuaternion start_ori(0, 0, 0, 1);
        sim->resetBasePositionAndOrientation(agent_id, start_pos, start_ori);
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        return get_observation();
    }

    // Takes a step in the simulation based on the given action (forces).
    // The actions tensor should have 3 elements: [force_x, force_y, force_z].
    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        // Apply forces to the agent's base link.
        float force_x = actions[0].item<float>();
        float force_y = actions[1].item<float>();
        float force_z = actions[2].item<float>();

        // We apply a scale factor to the forces to make the simulation more stable.
        btVector3 force(force_x * 10.0f, force_y * 10.0f, force_z * 10.0f);
        btVector3 pos_in_link_frame(0, 0, 0); // Apply force at the center of mass
        int link_index = -1; // -1 for the base link

        sim->applyExternalForce(agent_id, link_index, force, pos_in_link_frame, EF_LINK_FRAME);

        // Step the simulation.
        sim->stepSimulation();

        // Get the agent's position to calculate the reward.
        btVector3 base_pos;
        btQuaternion base_ori;
        sim->getBasePositionAndOrientation(agent_id, base_pos, base_ori);

        // Calculate the distance to the target point (0, 0, 2).
        btVector3 target_pos(0.0f, 0.0f, 3.0f);
        btVector3 dist_vec = base_pos - target_pos;
        float distance = dist_vec.length();

        // Reward is based on closeness to the target, with a penalty for large forces.
        float force_magnitude_sq = force_x * force_x + force_y * force_y + force_z * force_z;
        float reward = -distance - 0.001f * force_magnitude_sq;

        // The episode is never "done".
        bool done = distance > 10 ? true : false;

        GetFps();

        return { get_observation(), reward, done, false };
    }

    // Defines the observation space.
    // We observe the base link's world position, orientation, linear velocity, and angular velocity.
    int observation_space() const override {
        // 3 pos + 4 ori (quaternion) + 3 linear velocity + 3 angular velocity = 13
        return 6;
    }

    // Defines the action space.
    // We have 3 actions for force control (X, Y, and Z).
    int action_space() const override {
        return 3;
    }

    // Gathers the current state of the environment into a tensor.
    torch::Tensor get_observation() override {
        std::vector<float> obs;

        // Get the agent's base state.
        btVector3 base_pos;
        btQuaternion base_ori;
        btVector3 linear_vel;
        btVector3 angular_vel;
        sim->getBasePositionAndOrientation(agent_id, base_pos, base_ori);
        sim->getBaseVelocity(agent_id, linear_vel, angular_vel);

        // Agent's world position.
        obs.push_back(static_cast<float>(base_pos[0]));
        obs.push_back(static_cast<float>(base_pos[1]));
        obs.push_back(static_cast<float>(base_pos[2]));

        // Agent's world linear velocity.
        obs.push_back(static_cast<float>(linear_vel[0]));
        obs.push_back(static_cast<float>(linear_vel[1]));
        obs.push_back(static_cast<float>(linear_vel[2]));

        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    // Renders the environment (currently a no-op).
    void render() override {}
    void animate() override {}
    void EnableManipulator() override {}
};