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
#include <algorithm>
#include <btBulletCollisionCommon.h>
#include <thread>
#include <chrono>
#include "RobotSimulator.h"
#include <tinyxml/tinyxml2.h>
#include <sys/stat.h>
#include "Manipulator.h"
// Include individual environment headers

// Abstract environment interface
class Env {
public:
    torch::Device& mDevice;
    Env(torch::Device& device) : mDevice(device) {}

    virtual ~Env() = default;

    struct Space {
        std::vector<int> shape;
    };

    virtual torch::Tensor reset(int index = -1) = 0;
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions, int frame_index) = 0;
    virtual void render() = 0;
    virtual void animate() = 0;
    virtual Space observation_space() const = 0;
    virtual Space action_space() const = 0;
    virtual void EnableManipulator() = 0;
    int GetGridCount() const { return (grid_size * grid_size); }
    int grid_size = 1;
    float grid_space = 20.0f;
};

class Env3D : public Env {
public:
    Env3D(torch::Device& device, b3RobotSimulatorClientAPI* sim_ptr) : Env(device), sim(sim_ptr)
    {
        if (!sim->connect(eCONNECT_GUI)) {
            printf("Cannot connect\n");
            return;
        }

        sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
        sim->setTimeOut(10);
        sim->syncBodies();
        sim->setTimeStep(1. / 240.);
        sim->setGravity(btVector3(0, 0, -9.8));
    }
    virtual ~Env3D() override = default;

    b3RobotSimulatorClientAPI* sim;
    std::vector<int> agent_ids;
    std::vector<int> target_ids;
    /// ANIMATION CODE
    int selected_joint_index = 0; // Track selected joint
    int selected_object_id = 0; // Track selected humanoid
    int current_animation_frame = 0; // Track current animation frame

    btVector3 start_pos = btVector3(0, 0, 0);
    btQuaternion start_ori = btQuaternion(0, 0, 0, 1);

    tinyxml2::XMLDocument animation_doc;

    Manipulator* manipulator;
    std::vector<double> joint_positions;

    void EnableManipulator() override
    {
        HINSTANCE hInstance = GetModuleHandle(NULL);
        int nCmdShow = SW_SHOWNORMAL;

        try {
            if (!sim || agent_ids.empty()) {
                MessageBoxA(NULL, "Simulator not initialized or no agents loaded. Cannot enable manipulator.", "Error", MB_ICONERROR);
                return;
            }

            int object_id = agent_ids[0]; // Assuming agent_ids[0] holds the unique ID of the robot

            int numJoints = sim->getNumJoints(object_id);

            if (numJoints <= 0) {
                MessageBoxA(NULL, "No joints found for the agent. Manipulator not enabled.", "Info", MB_OK | MB_ICONINFORMATION);
                return;
            }

            // Collect joint names
            std::vector<std::string> jointNames;
            for (int j = 0; j < numJoints; ++j) {
                b3JointInfo jointInfo;
                sim->getJointInfo(object_id, j, &jointInfo);
                jointNames.push_back(jointInfo.m_jointName); // Assuming m_jointName is a char* or std::string
            }

            // Pass joint names to the Manipulator constructor instead of just the count
            manipulator = new Manipulator(hInstance, 100, 100, 400, 350 + numJoints * 40, "Robot Arm Control", jointNames);
            manipulator->Show(nCmdShow);

            // Set default position ranges (-1.0 to +1.0 with 0.1 increment, so -10 to 10 scaled)
            manipulator->SetPositionRange(0, -1.0, 1.0); // X
            manipulator->SetPositionRange(1, -1.0, 1.0); // Y
            manipulator->SetPositionRange(2, -1.0, 1.0); // Z

            // Set default rotation ranges (e.g., -180 to 180 degrees)
            manipulator->SetRotationRange(0, -180, 180); // Roll
            manipulator->SetRotationRange(1, -180, 180); // Pitch
            manipulator->SetRotationRange(2, -180, 180); // Yaw

            for (int j = 0; j < numJoints; ++j) {
                b3JointInfo jointInfo;
                sim->getJointInfo(object_id, j, &jointInfo);

                const double RAD_TO_DEG = 180.0 / SIMD_PI;

                int minAngle = static_cast<int>(jointInfo.m_jointLowerLimit * RAD_TO_DEG);
                int maxAngle = static_cast<int>(jointInfo.m_jointUpperLimit * RAD_TO_DEG);

                if (minAngle >= maxAngle) {
                    minAngle = -180;
                    maxAngle = 180;
                }
                manipulator->SetJointRange(j, minAngle, maxAngle);

                // Optional: Initialize joint_positions with current joint states
                b3JointSensorState state;
                sim->getJointState(object_id, j, &state);
                manipulator->SetCurrentJointAngles({ static_cast<int>(state.m_jointPosition * RAD_TO_DEG) }); // Update UI
            }

            manipulator->SetFrameRange(0, 500); // Example: 0 to 500 frames

            // Set the callback for manipulator changes and button clicks
            manipulator->SetOnManipulatorChangeCallback(
                [&, object_id, numJoints](ManipulatorAction action,
                    const std::vector<int>& positionXYZ_scaled,
                    const std::vector<int>& rotationXYZ_degrees,
                    const std::vector<int>& jointAngles_degrees,
                    int frameNumber) {
                        switch (action) {
                        case ManipulatorAction::SLIDER_CHANGE: {
                            std::cout << "Position: [" << static_cast<double>(positionXYZ_scaled[0]) / 10.0 << " "
                                << static_cast<double>(positionXYZ_scaled[1]) / 10.0 << " "
                                << static_cast<double>(positionXYZ_scaled[2]) / 10.0 << "], ";
                            std::cout << "Rotation: [" << rotationXYZ_degrees[0] << " " << rotationXYZ_degrees[1] << " " << rotationXYZ_degrees[2] << "], ";
                            std::cout << "Joint Angles: [";
                            for (int angle : jointAngles_degrees) {
                                std::cout << angle << " ";
                            }
                            std::cout << "], Frame: " << frameNumber << std::endl;

                            // Apply Base Position and Orientation
                            btVector3 start_pos(static_cast<double>(positionXYZ_scaled[0]) / 10.0,
                                static_cast<double>(positionXYZ_scaled[1]) / 10.0,
                                static_cast<double>(positionXYZ_scaled[2]) / 10.0);

                            const double DEG_TO_RAD = SIMD_PI / 180.0;
                            btQuaternion start_ori;
                            start_ori.setEulerZYX(static_cast<double>(rotationXYZ_degrees[2]) * DEG_TO_RAD, // Z (Yaw)
                                static_cast<double>(rotationXYZ_degrees[1]) * DEG_TO_RAD, // Y (Pitch)
                                static_cast<double>(rotationXYZ_degrees[0]) * DEG_TO_RAD); // X (Roll)

                            sim->resetBasePositionAndOrientation(object_id, start_pos, start_ori);

                            // Apply Joint Angles
                            for (int j = 0; j < numJoints; ++j) {
                                if (j < jointAngles_degrees.size()) {
                                    double targetPositionRad = static_cast<double>(jointAngles_degrees[j]) * DEG_TO_RAD;

                                    b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                                    motorArgs.m_targetPosition = targetPositionRad;
                                    motorArgs.m_targetVelocity = 1.0;
                                    motorArgs.m_kp = 0.1;
                                    motorArgs.m_kd = 0.5;
                                    motorArgs.m_maxTorqueValue = 1000.0;

                                    sim->setJointMotorControl(object_id, j, motorArgs);
                                }
                            }
                            break;
                        }
                        case ManipulatorAction::SAVE: {
                            std::cout << "Save button clicked for frame: " << frameNumber << std::endl;
                            saveRobotStateToXml(object_id, frameNumber, positionXYZ_scaled, rotationXYZ_degrees);
                            break;
                        }
                        case ManipulatorAction::LOAD: {
                            std::cout << "Load button clicked for frame: " << frameNumber << std::endl;
                            loadRobotStateFromXml(object_id, frameNumber);
                            break;
                        }
                        }
                }
            );

        }
        catch (const std::runtime_error& e) {
            MessageBoxA(NULL, e.what(), "Error", MB_ICONERROR);
            return;
        }
    }

    void saveRobotStateToXml(int object_id, int frame_number, const std::vector<int>& positionXYZ_scaled, const std::vector<int>& rotationXYZ_degrees) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to save.\n");
                return;
            }

            // Ensure animations directory exists
            std::filesystem::create_directories("animations");

            tinyxml2::XMLElement* root = animation_doc.FirstChildElement("Animation");
            if (!root) {
                root = animation_doc.NewElement("Animation");
                animation_doc.InsertEndChild(root);
            }

            tinyxml2::XMLElement* frameElem = nullptr;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                if (elem->IntAttribute("index") == frame_number) {
                    frameElem = elem;
                    break;
                }
            }

            if (!frameElem) {
                // If frame does not exist, create a new one and append it
                frameElem = animation_doc.NewElement("Frame");
                frameElem->SetAttribute("index", frame_number);
                root->InsertEndChild(frameElem); // Insert the new frame directly
            }
            else {
                // If frame exists, clear its children to update its content
                frameElem->DeleteChildren();
            }

            // Save base position
            tinyxml2::XMLElement* posElem = animation_doc.NewElement("BasePosition");
            posElem->SetAttribute("x", static_cast<double>(positionXYZ_scaled[0]) / 10.0);
            posElem->SetAttribute("y", static_cast<double>(positionXYZ_scaled[1]) / 10.0);
            posElem->SetAttribute("z", static_cast<double>(positionXYZ_scaled[2]) / 10.0);
            frameElem->InsertEndChild(posElem);

            // Save base rotation (Euler angles in degrees)
            tinyxml2::XMLElement* rotElem = animation_doc.NewElement("BaseRotation");
            rotElem->SetAttribute("roll", rotationXYZ_degrees[0]);
            rotElem->SetAttribute("pitch", rotationXYZ_degrees[1]);
            rotElem->SetAttribute("yaw", rotationXYZ_degrees[2]);
            frameElem->InsertEndChild(rotElem);

            // Save joint positions
            int num_joints = sim->getNumJoints(object_id);
            for (int j = 0; j < num_joints; ++j) {
                b3JointSensorState state;
                if (sim->getJointState(object_id, j, &state)) {
                    tinyxml2::XMLElement* jointElem = animation_doc.NewElement("Joint");
                    jointElem->SetAttribute("index", j);
                    jointElem->SetAttribute("position", state.m_jointPosition); // Store in radians
                    frameElem->InsertEndChild(jointElem);
                }
                else {
                    printf("Warning: Could not get state for joint %d.\n", j);
                }
            }

            // No sorting needed, just save the document
            animation_doc.SaveFile("animations/animation.xml");
            printf("Robot state saved to animations/animation.xml for frame %d.\n", frame_number);

        }
        catch (const std::exception& e) {
            printf("Exception in saveRobotStateToXml: %s\n", e.what());
        }
        catch (...) {
            printf("Unknown exception in saveRobotStateToXml.\n");
        }
    }

    // Function to load robot state from XML
    void loadRobotStateFromXml(int object_id, int frame_number) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to load state for.\n");
                return;
            }

            tinyxml2::XMLError err = animation_doc.LoadFile("animations/animation.xml");
            if (err != tinyxml2::XML_SUCCESS) {
                printf("Failed to load animation.xml: %s\n", animation_doc.ErrorStr());
                return;
            }

            tinyxml2::XMLElement* root = animation_doc.FirstChildElement("Animation");
            if (!root) {
                printf("Animation root element not found in XML.\n");
                return;
            }

            tinyxml2::XMLElement* frameElem = nullptr;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                if (elem->IntAttribute("index") == frame_number) {
                    frameElem = elem;
                    break;
                }
            }

            if (!frameElem) {
                printf("Frame %d not found in animation.xml.\n", frame_number);
                return;
            }

            // Load base position
            tinyxml2::XMLElement* posElem = frameElem->FirstChildElement("BasePosition");
            btVector3 loaded_pos(0, 0, 0);
            if (posElem) {
                loaded_pos.setX(posElem->DoubleAttribute("x"));
                loaded_pos.setY(posElem->DoubleAttribute("y"));
                loaded_pos.setZ(posElem->DoubleAttribute("z"));
            }
            else {
                printf("Warning: BasePosition not found for frame %d.\n", frame_number);
            }

            // Load base rotation (Euler angles in degrees)
            tinyxml2::XMLElement* rotElem = frameElem->FirstChildElement("BaseRotation");
            btQuaternion loaded_ori;
            if (rotElem) {
                double roll_deg = rotElem->DoubleAttribute("roll");
                double pitch_deg = rotElem->DoubleAttribute("pitch");
                double yaw_deg = rotElem->DoubleAttribute("yaw");
                const double DEG_TO_RAD = SIMD_PI / 180.0;
                loaded_ori.setEulerZYX(yaw_deg * DEG_TO_RAD, pitch_deg * DEG_TO_RAD, roll_deg * DEG_TO_RAD);
            }
            else {
                printf("Warning: BaseRotation not found for frame %d.\n", frame_number);
            }

            sim->resetBasePositionAndOrientation(object_id, loaded_pos, loaded_ori);

            // Update manipulator UI with loaded base position/rotation
            std::vector<double> pos_vec = { loaded_pos.x(), loaded_pos.y(), loaded_pos.z() };
            manipulator->SetCurrentPosition(pos_vec);

            // Convert quaternion back to Euler for UI (approximate, for display)
            btMatrix3x3 mat(loaded_ori);
            btScalar roll_rad_bt, pitch_rad_bt, yaw_rad_bt; // Declare as btScalar
            mat.getEulerZYX(yaw_rad_bt, pitch_rad_bt, roll_rad_bt); // Use btScalar variables
            const double RAD_TO_DEG = 180.0 / SIMD_PI;
            manipulator->SetCurrentRotation({ static_cast<int>(roll_rad_bt * RAD_TO_DEG),
                                             static_cast<int>(pitch_rad_bt * RAD_TO_DEG),
                                             static_cast<int>(yaw_rad_bt * RAD_TO_DEG) });

            // Load joint positions
            std::vector<int> loaded_joint_angles_degrees;
            int num_joints = sim->getNumJoints(object_id);
            loaded_joint_angles_degrees.resize(num_joints);

            for (tinyxml2::XMLElement* jointElem = frameElem->FirstChildElement("Joint"); jointElem; jointElem = jointElem->NextSiblingElement("Joint")) {
                int index = jointElem->IntAttribute("index");
                double position_rad = jointElem->DoubleAttribute("position");
                if (index >= 0 && index < num_joints) {
                    b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                    motorArgs.m_targetPosition = position_rad;
                    motorArgs.m_targetVelocity = 1.0;
                    motorArgs.m_kp = 0.1;
                    motorArgs.m_kd = 0.5;
                    motorArgs.m_maxTorqueValue = 1000.0;
                    sim->setJointMotorControl(object_id, index, motorArgs);

                    loaded_joint_angles_degrees[index] = static_cast<int>(position_rad * RAD_TO_DEG);
                }
            }
            manipulator->SetCurrentJointAngles(loaded_joint_angles_degrees);
            manipulator->SetCurrentFrameNumber(frame_number); // Update frame number in UI

            printf("Robot state loaded from animations/animation.xml for frame %d.\n", frame_number);

        }
        catch (const std::exception& e) {
            printf("Exception in loadRobotStateFromXml: %s\n", e.what());
        }
        catch (...) {
            printf("Unknown exception in loadRobotStateFromXml.\n");
        }
    }

    // Remove anim_skip_steps argument from animate
    void animate() override {
        // Mouse interaction for joint manipulation

        static btVector3 last_mouse_pos;

        printf("Animation mode started. Press ESC to exit.\n");
        printf("Press 'S' to save joint positions, 'R' to reset to saved positions.\n");
        printf("Use 1/2 to select joint, 3/4 to modify position.\n");
        printf("Use 5/6 to select animation frame.\n");

        // Run animation loop until escape is pressed
        using clock = std::chrono::steady_clock;
        auto last_event_time = clock::now();
        while (true && agent_ids.size() > 0) {
            //sim->resetBasePositionAndOrientation(agent_ids[0], start_pos, start_ori);
            sim->resetBaseVelocity(agent_ids[0], btVector3(0, 0, 0), btVector3(0, 0, 0));

            // Step simulation to apply joint changes
            sim->stepSimulation();

            // Main message loop (if you're not using GraphWindowManager's implicit loop)
            MSG msg = {};
            if(GetMessage(&msg, NULL, 0, 0)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }

    // External function to save joint positions
    void saveJointPositions(int frame = -1) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to save.\n");
                return;
            }
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            std::vector<double> joint_positions;
            for (int j = 0; j < num_joints; ++j) {
                b3JointSensorState state;
                if (sim->getJointState(object_id, j, &state)) {
                    joint_positions.push_back(state.m_jointPosition);
                } else {
                    joint_positions.push_back(0.0);
                }
            }

            // Ensure animations directory exists
            std::filesystem::create_directories("animations");

            // Use cached animation_doc
            tinyxml2::XMLElement* root = animation_doc.FirstChildElement("Animation");
            if (!root) {
                root = animation_doc.NewElement("Animation");
                animation_doc.InsertEndChild(root);
            }

            int frame_index = (frame >= 0) ? frame : current_animation_frame;
            // Find or create frame element
            tinyxml2::XMLElement* frameElem = nullptr;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                if (elem->IntAttribute("index") == frame_index) {
                    frameElem = elem;
                    break;
                }
            }
            if (!frameElem) {
                frameElem = animation_doc.NewElement("Frame");
                frameElem->SetAttribute("index", frame_index);
                root->InsertEndChild(frameElem);
            } else {
                frameElem->DeleteChildren(); // Clear previous joint data
            }

            // Save joint positions
            for (int j = 0; j < num_joints; ++j) {
                tinyxml2::XMLElement* jointElem = animation_doc.NewElement("Joint");
                jointElem->SetAttribute("index", j);
                jointElem->SetAttribute("position", joint_positions[j]);
                frameElem->InsertEndChild(jointElem);
            }

            // --- Sort <Frame> elements by 'index' attribute before saving ---
            std::vector<tinyxml2::XMLElement*> frames;
            for (tinyxml2::XMLElement* elem = root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
                frames.push_back(elem);
            }
            std::sort(frames.begin(), frames.end(), [](tinyxml2::XMLElement* a, tinyxml2::XMLElement* b) {
                return a->IntAttribute("index") < b->IntAttribute("index");
            });
            for (tinyxml2::XMLElement* elem : frames) {
                root->InsertEndChild(elem);
            }
            // --- End sorting ---

            animation_doc.SaveFile("animations/animation.xml");
        } catch (const std::exception& e) {
            printf("Exception in saveJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in saveJointPositions.\n");
        }
    }

    // External function to reset joint positions
    void resetJointPositions(int frame = -1) {
        try {
            if (agent_ids.empty()) {
                printf("No humanoids available to reset.\n");
                return;
            }
            int object_id = agent_ids[0]; // Only one humanoid
            int num_joints = sim->getNumJoints(object_id);
            std::vector<double> joint_positions = GetJointAnim(frame);
            for (int j = 0; j < num_joints; ++j) {
                b3RobotSimulatorJointMotorArgs motorArgs(CONTROL_MODE_POSITION_VELOCITY_PD);
                motorArgs.m_targetPosition = joint_positions[j];
                motorArgs.m_targetVelocity = 1.0; // Large value for instant movement
                motorArgs.m_maxTorqueValue = 100.0; // Large torque for instant movement
                sim->setJointMotorControl(object_id, j, motorArgs);
            }
        } catch (const std::exception& e) {
            printf("Exception in resetJointPositions: %s\n", e.what());
        } catch (...) {
            printf("Unknown exception in resetJointPositions.\n");
        }
    }

    // Get joint animation for the current or last frame
    std::vector<double> GetJointAnim(int frame = -1) {
        std::vector<double> joint_positions;
        tinyxml2::XMLElement* animation_root = animation_doc.FirstChildElement("Animation");
        int num_joints = 0;
        if (!agent_ids.empty()) {
            int object_id = agent_ids[0];
            num_joints = sim->getNumJoints(object_id);
        }
        if (animation_doc.ErrorID() != 0 || !animation_root) {
            // No animation file: return zeros
            return std::vector<double>(num_joints, 0.0);
        }
        // Find the frame with the given frame number, or last if frame == -1
        tinyxml2::XMLElement* frameElem = nullptr;
        int last_index = -1;
        tinyxml2::XMLElement* lastElem = nullptr;
        for (tinyxml2::XMLElement* elem = animation_root->FirstChildElement("Frame"); elem; elem = elem->NextSiblingElement("Frame")) {
            int idx = elem->IntAttribute("index");
            if ((frame >= 0 && idx == frame)) {
                frameElem = elem;
                break;
            }
            if (idx > last_index) {
                last_index = idx;
                lastElem = elem;
            }
        }
        if (!frameElem && lastElem) frameElem = lastElem;
        if (!frameElem) return std::vector<double>(num_joints, 0.0);
        // Read joint positions
        for (tinyxml2::XMLElement* jointElem = frameElem->FirstChildElement("Joint"); jointElem; jointElem = jointElem->NextSiblingElement("Joint")) {
            joint_positions.push_back(jointElem->DoubleAttribute("position"));
        }
        // If mismatch, return zeros
        if (joint_positions.size() != static_cast<size_t>(num_joints)) {
            return std::vector<double>(num_joints, 0.0);
        }
        return joint_positions;
    }
};
