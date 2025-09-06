#pragma once
#include "RobotSimulator.h"
#include "../env.h"
#include <vector>
#include <stdexcept>
#include <string>

// A 3D environment for simulating a double pendulum with two degrees of freedom.
class Pendulum3dEnv : public Env3D {
public:
    // Constructor. Loads the URDF and sets up the environment.
    Pendulum3dEnv(torch::Device& device)
        : Env3D(device, new b3RobotSimulatorClientAPI())
    {
        // Load a plane for the ground.
        b3RobotSimulatorLoadUrdfFileArgs plane_args;
        plane_args.m_startPosition = { 0.0f, 0.0f, 0.0f };
        sim->loadURDF("plane.urdf", plane_args);

        // Load the double pendulum URDF. We assume this URDF has two joints named
        // "pendulum_x" and "pendulum_y".
        b3RobotSimulatorLoadUrdfFileArgs pendulum_args;
        pendulum_args.m_startPosition = { 0.0f, 0.0f, 2.0f }; // Start elevated at Z = 2.0f
        pendulum_args.m_useMultiBody = true;
        pendulum_args.m_flags = 0;
        agent_id = sim->loadURDF("pendulum.urdf", pendulum_args);

        // Find the IDs for the two pendulum joints.
        int num_joints = sim->getNumJoints(agent_id);
        pendulum_x_joint_id = -1;
        pendulum_y_joint_id = -1;
        for (int i = 0; i < num_joints; ++i) {
            b3JointInfo jointInfo;
            if (sim->getJointInfo(agent_id, i, &jointInfo)) {
                if (std::string(jointInfo.m_jointName) == "pendulum_x") {
                    pendulum_x_joint_id = i;
                }
                if (std::string(jointInfo.m_jointName) == "pendulum_y") {
                    pendulum_y_joint_id = i;
                }
            }
        }

        // Throw an exception if one or both joints are not found.
        if (pendulum_x_joint_id == -1 || pendulum_y_joint_id == -1) {
            throw std::runtime_error("Could not find both 'pendulum_x' and 'pendulum_y' joints in the URDF.");
        }

        // Set simulation parameters.
        sim->setGravity({ 0, 0, -9.8 });
        sim->setTimeStep(1.0 / 240.0);
        sim->setRealTimeSimulation(false);
    }

    // Resets the pendulum to its initial upright position and elevated state.
    torch::Tensor reset() override {
        // Reset the joint states to 0 position and velocity.
        sim->resetJointState(agent_id, pendulum_x_joint_id, 0.0);
        sim->resetJointState(agent_id, pendulum_y_joint_id, 0.0);

        // Reset base state to ensure consistency, though it's fixed in the URDF.
        btVector3 start_pos(0.0f, 0.0f, 2.0f);
        btQuaternion start_ori(0, 0, 0, 1);
        sim->resetBasePositionAndOrientation(agent_id, start_pos, start_ori);
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        return get_observation();
    }

    // Takes a step in the simulation based on the given action (torque).
    // The actions tensor should have 2 elements: [torque_x, torque_y].
    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        // Keep the pendulum elevated at 2.0 on the Z axis on every step
        btVector3 start_pos(0.0f, 0.0f, 2.0f);
        btQuaternion start_ori(0, 0, 0, 1);
        sim->resetBasePositionAndOrientation(agent_id, start_pos, start_ori);
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        // Apply torques to the joints.
        float torque_x = actions[0].item<float>() * 100.0f;
        float torque_y = actions[1].item<float>() * 100.0f;

        b3RobotSimulatorJointMotorArgs args_x(CONTROL_MODE_TORQUE);
        args_x.m_maxTorqueValue = torque_x;
        sim->setJointMotorControl(agent_id, pendulum_x_joint_id, args_x);

        b3RobotSimulatorJointMotorArgs args_y(CONTROL_MODE_TORQUE);
        args_y.m_maxTorqueValue = torque_y;
        sim->setJointMotorControl(agent_id, pendulum_y_joint_id, args_y);

        // Step the simulation.
        sim->stepSimulation();

        // Get the state of the pendulum link to calculate the reward.
        b3LinkState link_state;
        // The pendulum_link is at index 0 in a 1-joint URDF.
        sim->getLinkState(agent_id, 0, 1, 0, &link_state);

        // Reward is based on the pendulum's Z-position.
        float z_pos = link_state.m_worldPosition[2];
        float reward = z_pos - 0.001f * (torque_x * torque_x + torque_y * torque_y);

        // The episode is never "done".
        bool done = false;

        GetFps();

        return { get_observation(), reward, done, false };
    }

    // Defines the observation space.
    // We observe the link's world position, orientation, linear velocity, and angular velocity,
    // as well as the states of both joints.
    Space observation_space() const override {
        // 3 pos + 4 ori (quaternion) + 3 linear velocity + 3 angular velocity + 2 joint angles + 2 joint velocities = 17
        return Space{ {17} };
    }

    // Defines the action space.
    // We have 2 actions for torque control (X and Y).
    Space action_space() const override {
        return Space{ {2} };
    }

    // Gathers the current state of the environment into a tensor.
    torch::Tensor get_observation() override {
        std::vector<float> obs;

        // Get the state of the pendulum link.
        b3LinkState link_state;
        // The pendulum_link is at index 0 in a 1-joint URDF.
        sim->getLinkState(agent_id, 0, 1, 0, &link_state);

        // Link world position.
        obs.push_back(static_cast<float>(link_state.m_worldPosition[0]));
        obs.push_back(static_cast<float>(link_state.m_worldPosition[1]));
        obs.push_back(static_cast<float>(link_state.m_worldPosition[2]));

        // Link world orientation (quaternion).
        obs.push_back(static_cast<float>(link_state.m_worldOrientation[0]));
        obs.push_back(static_cast<float>(link_state.m_worldOrientation[1]));
        obs.push_back(static_cast<float>(link_state.m_worldOrientation[2]));
        obs.push_back(static_cast<float>(link_state.m_worldOrientation[3]));

        // Link world linear velocity.
        obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[0]));
        obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[1]));
        obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[2]));

        // Link world angular velocity.
        obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[0]));
        obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[1]));
        obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[2]));

        // Get the state of the two joints.
        b3JointSensorState joint_state_x;
        sim->getJointState(agent_id, pendulum_x_joint_id, &joint_state_x);
        obs.push_back(static_cast<float>(joint_state_x.m_jointPosition));
        obs.push_back(static_cast<float>(joint_state_x.m_jointVelocity));

        b3JointSensorState joint_state_y;
        sim->getJointState(agent_id, pendulum_y_joint_id, &joint_state_y);
        obs.push_back(static_cast<float>(joint_state_y.m_jointPosition));
        obs.push_back(static_cast<float>(joint_state_y.m_jointVelocity));

        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    // Renders the environment (currently a no-op).
    void render() override {}
    void animate() override {}
    void EnableManipulator() override {}

private:
    int pendulum_x_joint_id; // ID for the pendulum's X-axis joint.
    int pendulum_y_joint_id; // ID for the pendulum's Y-axis joint.
};