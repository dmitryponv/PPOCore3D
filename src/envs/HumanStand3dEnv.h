#pragma once
#include "../env.h"

class HumanstandEnv : public Env3D {
private:
    // std::vector<std::vector<b3LinkState>> saved_link_states; // Remove unused
    // std::vector<btVector3> saved_base_positions; // Remove unused
    // std::vector<btQuaternion> saved_base_orientations; // Remove unused

public:
    HumanstandEnv(torch::Device& device)
        : Env3D(device, new b3RobotSimulatorClientAPI()) // Pass sim pointer to base
    {
        start_ori.setEulerZYX(0, 0, 0); // 90 degrees around Y-axis
        start_pos = { 0,0,1.0 };
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

        agent_id = sim->loadURDF("goat.urdf", args);
        sim->setRealTimeSimulation(false);
    }

    Space observation_space() const override {
        int num_joints = sim->getNumJoints(agent_id);
        return Space{ {13 + num_joints * 21} };
    }

    Space action_space() const override {
        return Space{ {sim->getNumJoints(agent_id)} };
    }

    torch::Tensor reset() override {
        btVector3 start_pos(0.0f, 0.0f, start_pos.z());

        sim->resetBasePositionAndOrientation(agent_id, start_pos, start_ori);
        sim->resetBaseVelocity(agent_id, btVector3(0, 0, 0), btVector3(0, 0, 0));

        int num_joints = sim->getNumJoints(agent_id);
        for (int k = 0; k < num_joints; ++k) {
            sim->resetJointState(agent_id, k, 0.0);
        }
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {

        // Get animation for this frame
        int num_joints = sim->getNumJoints(agent_id);
        const float max_velocity = 1.0f;
        // bool done = false; // done is determined after sim->stepSimulation()

        for (int j = 0; j < num_joints; ++j) {
            b3JointInfo jointInfo;
            sim->getJointInfo(agent_id, j, &jointInfo);

            if (jointInfo.m_jointType != JointType::eRevoluteType) {
                continue;
            }

            float action_single = actions[j].item<float>()*10.0f;

            b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
            motorArgs.m_maxTorqueValue = 200.0f;
            motorArgs.m_targetPosition = action_single;
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

        //FOR TESTING ANIMATIONS
        //sim->resetBasePositionAndOrientation(id, start_pos, start_ori); sim->resetBaseVelocity(id, btVector3(0, 0, 0), btVector3(0, 0, 0));

    }

    torch::Tensor get_observation() {
        std::vector<float> obs;

        // 1. Get observations for the entire body (base link)
        btVector3 base_position;     // Will store the position
        btQuaternion base_orientation; // Will store the orientation
        btVector3 base_linear_vel;   // Will store the linear velocity
        btVector3 base_angular_vel;  // Will store the angular velocity

        // Use the specified function signature
        // Pass references to btVector3 and btQuaternion objects
        sim->getBasePositionAndOrientation(agent_id, base_position, base_orientation);
        sim->getBaseVelocity(agent_id, base_linear_vel, base_angular_vel);

        // Base position
        obs.push_back(static_cast<float>(base_position.getX()));
        obs.push_back(static_cast<float>(base_position.getY()));
        obs.push_back(static_cast<float>(base_position.getZ()));

        // Base orientation (quaternion)
        obs.push_back(static_cast<float>(base_orientation.getX())); // x component of quaternion
        obs.push_back(static_cast<float>(base_orientation.getY())); // y component of quaternion
        obs.push_back(static_cast<float>(base_orientation.getZ())); // z component of quaternion
        obs.push_back(static_cast<float>(base_orientation.getW())); // w component of quaternion

        // Base linear velocity
        obs.push_back(static_cast<float>(base_linear_vel.getX()));
        obs.push_back(static_cast<float>(base_linear_vel.getY()));
        obs.push_back(static_cast<float>(base_linear_vel.getZ()));

        // Base angular velocity
        obs.push_back(static_cast<float>(base_angular_vel.getX()));
        obs.push_back(static_cast<float>(base_angular_vel.getY()));
        obs.push_back(static_cast<float>(base_angular_vel.getZ()));

        // 2. Get observations for each link
        int num_joints = sim->getNumJoints(agent_id); // This also represents the number of links attached to joints
        for (int j = 0; j < num_joints; ++j) {
            b3LinkState link_state; // b3LinkState is likely from your PyBullet C++ API headers
            sim->getLinkState(agent_id, j, 1, 0, &link_state);
        
            obs.push_back(static_cast<float>(link_state.m_worldPosition[0]));
            obs.push_back(static_cast<float>(link_state.m_worldPosition[1]));
            obs.push_back(static_cast<float>(link_state.m_worldPosition[2]));
        
            obs.push_back(static_cast<float>(link_state.m_worldOrientation[0]));
            obs.push_back(static_cast<float>(link_state.m_worldOrientation[1]));
            obs.push_back(static_cast<float>(link_state.m_worldOrientation[2]));
            obs.push_back(static_cast<float>(link_state.m_worldOrientation[3]));
        
            obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[0]));
            obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[1]));
            obs.push_back(static_cast<float>(link_state.m_worldLinearVelocity[2]));
        
            obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[0]));
            obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[1]));
            obs.push_back(static_cast<float>(link_state.m_worldAngularVelocity[2]));
        }
        
        // 3. Get observations for each joint, still without m_jointReactionForces
        for (int j = 0; j < num_joints; ++j) {
            b3JointSensorState joint_state; // b3JointSensorState is likely from your PyBullet C++ API headers
            sim->getJointState(agent_id, j, &joint_state);
        
            // Joint position (angle for revolute/prismatic joints)
            obs.push_back(static_cast<float>(joint_state.m_jointPosition));
        
            // Joint velocity
            obs.push_back(static_cast<float>(joint_state.m_jointVelocity));
        
            // IMPORTANT: m_jointReactionForces is still removed because your previous error
            // indicated it's not a member of your b3JointSensorState.
            // If your headers *do* eventually have it, uncomment this.
            for (int k = 0; k < 6; ++k) {
                obs.push_back(static_cast<float>(joint_state.m_jointForceTorque[k]));
            }
        }

        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    void render() override {
        // No-op
    }
}; 