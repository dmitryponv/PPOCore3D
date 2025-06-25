#include "env.h"

RobotEnv::RobotEnv(torch::Device& device) : mDevice(device) {
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
    sim->loadURDF("plane.urdf");

    MinitaurSetup minitaur;
    minitaurUid = minitaur.setupMinitaur(sim, btVector3(0, 0, .3));
    numJoints = sim->getNumJoints(minitaurUid);

    // Precompute joints with valid axes for torque application
    for (int j = 0; j < numJoints; ++j) {
        b3JointInfo jointInfo;
        sim->getJointInfo(minitaurUid, j, &jointInfo);
        std::string jointName(jointInfo.m_linkName);
        btVector3 axis(jointInfo.m_jointAxis[0], jointInfo.m_jointAxis[1], jointInfo.m_jointAxis[2]);

        if (!axis.fuzzyZero() &&
            jointName.find("motor") != std::string::npos &&
            jointName.find("bracket") == std::string::npos) {
            validTorqueJoints.push_back(j);
        }
    }

    sim->setRealTimeSimulation(false);
}

Env::Space RobotEnv::observation_space() const {
    // Observations for all joints (positions + velocities) + base pos/vel
    return Space{ {6 + numJoints * 6} };
}

Env::Space RobotEnv::action_space() const {
    // Actions only for joints with non-zero axis
    return Space{ {static_cast<int>(validTorqueJoints.size())} };
}

std::pair<torch::Tensor, std::unordered_map<std::string, float>> RobotEnv::reset() {
    sim->resetBasePositionAndOrientation(minitaurUid, btVector3(0, 0, 0), btQuaternion(0, 0, 0, 1));
    sim->resetBaseVelocity(minitaurUid, btVector3(0, 0, 0), btVector3(0, 0, 0));

    for (int j = 0; j < numJoints; ++j) {
        sim->resetJointState(minitaurUid, j, 0.0);
    }

    return { get_observation(), {} };
}

std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> RobotEnv::step(const torch::Tensor& action) {
    static b3Clock clock;
    static double lastTime = clock.getTimeInSeconds();
    static int frameCount = 0;
    static int fpsTextId = -1;

    for (size_t i = 0; i < validTorqueJoints.size(); ++i) {
        int j = validTorqueJoints[i];
        float torque = action[i].item<float>() * 10.0f;
        torque = std::clamp(torque, -5.0f, 5.0f);
        b3JointInfo jointInfo;
        sim->getJointInfo(minitaurUid, j, &jointInfo);
        btVector3 axis(jointInfo.m_jointAxis[0], jointInfo.m_jointAxis[1], jointInfo.m_jointAxis[2]);
        btVector3 torqueVec = axis.normalized() * torque;

        sim->applyExternalTorque(minitaurUid, j, torqueVec, EF_LINK_FRAME);
    }

    sim->stepSimulation();

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

    b3LinkState baseLinkState;
    sim->getLinkState(minitaurUid, 0, 1, 1, &baseLinkState);
    float x = baseLinkState.m_worldPosition[0];
    float y = baseLinkState.m_worldPosition[1];
    float z = baseLinkState.m_worldPosition[2];

    float reward = z - 0.05f - 0.1f * std::sqrt(x * x + y * y);
    bool done = false;

    return { get_observation(), reward, done, false, {} };
}


void RobotEnv::render() {}

torch::Tensor RobotEnv::get_observation() const {
    std::vector<float> obs;

    // Base link state (link 0)
    b3LinkState baseLinkState;
    sim->getLinkState(minitaurUid, 0, 1, 1, &baseLinkState);
    obs.push_back(baseLinkState.m_worldPosition[0]);
    obs.push_back(baseLinkState.m_worldPosition[1]);
    obs.push_back(baseLinkState.m_worldPosition[2]);
    obs.push_back(baseLinkState.m_worldLinearVelocity[0]);
    obs.push_back(baseLinkState.m_worldLinearVelocity[1]);
    obs.push_back(baseLinkState.m_worldLinearVelocity[2]);

    // All joint link states
    for (int j = 0; j < numJoints; ++j) {
        b3LinkState linkState;
        sim->getLinkState(minitaurUid, j, 1, 1, &linkState);

        obs.push_back(linkState.m_worldPosition[0]);
        obs.push_back(linkState.m_worldPosition[1]);
        obs.push_back(linkState.m_worldPosition[2]);
        obs.push_back(linkState.m_worldLinearVelocity[0]);
        obs.push_back(linkState.m_worldLinearVelocity[1]);
        obs.push_back(linkState.m_worldLinearVelocity[2]);
    }

    return torch::from_blob(obs.data(), { (int64_t)obs.size() }).clone().to(mDevice);
}
