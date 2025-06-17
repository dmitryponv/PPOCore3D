#include "env.h"

RobotEnv::RobotEnv(torch::Device& device) : mDevice(device) {
    sim = new b3RobotSimulatorClientAPI();
    bool isConnected = sim->connect(eCONNECT_GUI);

    if (!isConnected)
    {
        printf("Cannot connect\n");
        return;
    };
    sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
    sim->setTimeOut(10);
    //syncBodies is only needed when connecting to an existing physics server that has already some bodies
    sim->syncBodies();
    btScalar fixedTimeStep = 1. / 240.;

    sim->setTimeStep(fixedTimeStep);

    btQuaternion q = sim->getQuaternionFromEuler(btVector3(0.1, 0.2, 0.3));
    btVector3 rpy;
    rpy = sim->getEulerFromQuaternion(q);

    sim->setGravity(btVector3(0, 0, -9.8));

    sim->loadURDF("plane.urdf");

    MinitaurSetup minitaur;
    int minitaurUid = minitaur.setupMinitaur(sim, btVector3(0, 0, .3));

    b3Clock clock;
    double startTime = clock.getTimeInSeconds();
    double simWallClockSeconds = 20.;

    sim->setRealTimeSimulation(false);
    int vidLogId = -1;
    int minitaurLogId = -1;
    int rotateCamera = 0;
}

Env::Space RobotEnv::observation_space() const{
    // Observations: agent_x, agent_y, target_x, target_y
    return Space{ {4} };
}

Env::Space RobotEnv::action_space() const{
    // Actions: continuous 2 floats, each in [-1, 1]
    return Space{ {2} };
}

std::pair<torch::Tensor, std::unordered_map<std::string, float>> RobotEnv::reset(){


    return { get_observation(), {} };
}

std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> RobotEnv::step(const torch::Tensor& action){
    sim->stepSimulation();

    // Reward and done conditions
    float reward = 1.0f;
    bool done = false;

    return { get_observation(), reward, done, false, {} };
}

void RobotEnv::render(){    

}

torch::Tensor RobotEnv::get_observation() const {
    return torch::tensor({ 0.0f,0.0f,0.0f,0.0f }).to(mDevice);
}