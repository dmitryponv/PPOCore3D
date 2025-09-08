#include <torch/torch.h>
#include <math.h>
#include <random>
#include <fstream>
#include <Eigen/Core>
#include <unordered_map>
#include <string>

// Forward declarations of external classes
class Env;
class GraphWindowManager;
class TestEnvironment;

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
static std::mt19937 re(std::random_device{}());

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module
{
    // Actor.
    torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
    torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.
    torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

    ActorCriticImpl(int64_t n_in, int64_t n_out, double std)
        : // Actor.
        a_lin1_(torch::nn::Linear(n_in, 16)),
        a_lin2_(torch::nn::Linear(16, 32)),
        a_lin3_(torch::nn::Linear(32, n_out)),
        mu_(torch::full(n_out, 0.)),
        log_std_(torch::full(n_out, std)),
        
        // Critic
        c_lin1_(torch::nn::Linear(n_in, 16)),
        c_lin2_(torch::nn::Linear(16, 32)),
        c_lin3_(torch::nn::Linear(32, n_out)),
        c_val_(torch::nn::Linear(n_out, 1))
    {
        // Register the modules.
        register_module("a_lin1", a_lin1_);
        register_module("a_lin2", a_lin2_);
        register_module("a_lin3", a_lin3_);
        register_parameter("log_std", log_std_);

        register_module("c_lin1", c_lin1_);
        register_module("c_lin2", c_lin2_);
        register_module("c_lin3", c_lin3_);
        register_module("c_val", c_val_);
    }

    // Forward pass.
    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor>
    {
        // Actor.
        mu_ = torch::relu(a_lin1_->forward(x));
        mu_ = torch::relu(a_lin2_->forward(mu_));
        mu_ = torch::tanh(a_lin3_->forward(mu_));

        // Critic.
        torch::Tensor val = torch::relu(c_lin1_->forward(x));
        val = torch::relu(c_lin2_->forward(val));
        val = torch::tanh(c_lin3_->forward(val));
        val = c_val_->forward(val);

        if (this->is_training())
        {
            torch::NoGradGuard no_grad;
            torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
            return std::make_tuple(action, val);
        }
        else
        {
            return std::make_tuple(mu_, val);
        }
    }

    // Initialize network.
    void normal(double mu, double std)
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters())
        {
            p.normal_(mu,std);
        }
    }

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution.
        return 0.5 + 0.5 * log(2 * M_PI) + log_std_;
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = (log_std_ + log_std_).exp();
        return -((action - mu_) * (action - mu_)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
    }
};

TORCH_MODULE(ActorCritic);

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO2
{
public:
    PPO2(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, GraphWindowManager& graph_manager, std::string actor_model, std::string critic_model)
        : mEnv(env),
          mHyperparameters(hyperparameters),
          mDevice(device),
          mGraphManager(graph_manager),
          mActorModel(actor_model),
          mCriticModel(critic_model),
          mAc(nullptr),
          mOpt(nullptr)
    {
        // Placeholder for initialization logic using constructor parameters.
        // The actual implementation would go here.
        std::cout << "PPO2 constructed with provided parameters." << std::endl;
    }

    // Main training function, refactored from the original main.
    void train(TestEnvironment& env)
    {
        // Model.
        uint n_in = 4;
        uint n_out = 2;
        double std = 2e-2;

        mAc = ActorCritic(n_in, n_out, std);
        mAc->to(torch::kF64);
        mAc->normal(0., std);
        mOpt = std::make_unique<torch::optim::Adam>(mAc->parameters(), 1e-3);

        // Training loop parameters
        uint n_iter = 10000;
        uint n_steps = 2048;
        uint n_epochs = 15;
        uint mini_batch_size = 512;
        uint ppo_epochs = 4;
        double beta = 1e-3;
        
        VT states;
        VT actions;
        VT rewards;
        VT dones;
        VT log_probs;
        VT returns;
        VT values;

        // Output.
        std::ofstream out;
        out.open("../data/data.csv");

        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
        // Note: RESETTING is a placeholder for a constant defined elsewhere.
        out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "RESETTING" << "\n";

        // Counter.
        uint c = 0;

        // Average reward.
        double best_avg_reward = 0.;
        double avg_reward = 0.;

        for (uint e=1; e<=n_epochs; e++)
        {
            printf("epoch %u/%u\n", e, n_epochs);

            for (uint i=0; i<n_iter; i++)
            {
                // State of env.
                states.push_back(env.State());

                // Play.
                auto av = mAc->forward(states[c]);
                actions.push_back(std::get<0>(av));
                values.push_back(std::get<1>(av));
                log_probs.push_back(mAc->log_prob(actions[c]));

                double x_act = actions[c][0][0].item<double>();
                double y_act = actions[c][0][1].item<double>();
                auto sd = env.Act(x_act, y_act);

                // New state.
                rewards.push_back(env.Reward(std::get<1>(sd)));
                dones.push_back(std::get<2>(sd));

                avg_reward += rewards[c][0][0].item<double>() / n_iter;

                // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
                out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "PLAYING" << "\n";

                if (dones[c][0][0].item<double>() == 1.)
                {
                    // Set new goal.
                    std::uniform_int_distribution<> dist(-5, 5);
                    double x_new = double(dist(re));
                    double y_new = double(dist(re));
                    env.SetGoal(x_new, y_new);

                    // Reset the position of the agent.
                    env.Reset();

                    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
                    out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "RESETTING" << "\n";
                }

                c++;

                // Update.
                if (c % n_steps == 0)
                {
                    printf("Updating the network.\n");
                    values.push_back(std::get<1>(mAc->forward(states[c-1])));

                    returns = returns_gae(rewards, dones, values, .99, .95);

                    torch::Tensor t_log_probs = torch::cat(log_probs).detach();
                    torch::Tensor t_returns = torch::cat(returns).detach();
                    torch::Tensor t_values = torch::cat(values).detach();
                    torch::Tensor t_states = torch::cat(states);
                    torch::Tensor t_actions = torch::cat(actions);
                    torch::Tensor t_advantages = t_returns - t_values.slice(0, 0, n_steps);

                    update_network(t_states, t_actions, t_log_probs, t_returns, t_advantages, n_steps, ppo_epochs, mini_batch_size, beta);
                    
                    c = 0;
                    states.clear();
                    actions.clear();
                    rewards.clear();
                    dones.clear();
                    log_probs.clear();
                    returns.clear();
                    values.clear();
                }
            }

            // Save the best net.
            if (avg_reward > best_avg_reward) {
                best_avg_reward = avg_reward;
                printf("Best average reward: %f\n", best_avg_reward);
                torch::save(mAc, "best_model.pt");
            }
            avg_reward = 0.;

            // Reset at the end of an epoch.
            std::uniform_int_distribution<> dist(-5, 5);
            double x_new = double(dist(re));
            double y_new = double(dist(re));
            env.SetGoal(x_new, y_new);

            // Reset the position of the agent.
            env.Reset();

            // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
            out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "RESETTING" << "\n";
        }
        out.close();
    }

    // Main test function, refactored from the original main.
    void test(TestEnvironment& env)
    {
        // Test loop.
        uint n_iter = 10000;
        
        // Output.
        std::ofstream out;
        out.open("../data/data_test.csv");

        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
        out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "RESETTING" << "\n";

        mAc->eval();
        torch::load(mAc, "best_model.pt");

        for (uint i=0; i<n_iter; i++)
        {
            // Play.
            auto av = mAc->forward(env.State());
            auto action = std::get<0>(av);

            double x_act = action[0][0].item<double>();
            double y_act = action[0][1].item<double>();
            auto sd = env.Act(x_act, y_act);

            // Check for done state.
            auto done = std::get<2>(sd);

            // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST, RESETTING)
            out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "PLAYING" << "\n";

            if (done[0][0].item<double>() == 1.)
            {
                // Set new goal.
                std::uniform_int_distribution<> dist(-5, 5);
                double x_new = double(dist(re));
                double y_new = double(dist(re));
                env.SetGoal(x_new, y_new);

                // Reset the position of the agent.
                env.Reset();

                // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
                out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << "RESETTING" << "\n";
            }
        }
        out.close();
    }

private:
    Env& mEnv;
    const std::unordered_map<std::string, float>& mHyperparameters;
    torch::Device& mDevice;
    GraphWindowManager& mGraphManager;
    std::string mActorModel;
    std::string mCriticModel;
    ActorCritic mAc;
    std::unique_ptr<torch::optim::Adam> mOpt;

    // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
    auto returns_gae(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
    {
        torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
        VT returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

        for (uint i=rewards.size(); i-- > 0;)
        {
            // Advantage.
            auto delta = rewards[i] + gamma * vals[i+1] * (1-dones[i]) - vals[i];
            gae = delta + gamma * lambda * (1-dones[i]) * gae;
            returns[i] = gae + vals[i];
        }
        return returns;
    }

    // Update the network.
    void update_network(torch::Tensor& states,
                        torch::Tensor& actions,
                        torch::Tensor& log_probs,
                        torch::Tensor& returns,
                        torch::Tensor& advantages,
                        uint steps, uint epochs, uint mini_batch_size, double beta, double clip_param = .2)
    {
        for (uint e=0; e<epochs; e++)
        {
            // Generate random indices.
            torch::Tensor cpy_sta = torch::zeros({mini_batch_size, states.size(1)}, states.options());
            torch::Tensor cpy_act = torch::zeros({mini_batch_size, actions.size(1)}, actions.options());
            torch::Tensor cpy_log = torch::zeros({mini_batch_size, log_probs.size(1)}, log_probs.options());
            torch::Tensor cpy_ret = torch::zeros({mini_batch_size, returns.size(1)}, returns.options());
            torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());

            for (uint b=0; b<mini_batch_size; b++) {
                uint idx = std::uniform_int_distribution<uint>(0, steps-1)(re);
                cpy_sta[b] = states[idx];
                cpy_act[b] = actions[idx];
                cpy_log[b] = log_probs[idx];
                cpy_ret[b] = returns[idx];
                cpy_adv[b] = advantages[idx];
            }

            auto av = mAc->forward(cpy_sta); // action value pairs
            auto entropy = mAc->entropy().mean();
            auto new_log_prob = mAc->log_prob(cpy_act);

            auto old_log_prob = cpy_log;
            auto ratio = (new_log_prob - old_log_prob).exp();
            auto surr1 = ratio * cpy_adv;
            auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;

            auto val = std::get<1>(av);
            auto actor_loss = -torch::min(surr1, surr2).mean();
            auto critic_loss = (cpy_ret - val).pow(2).mean();

            auto loss = 0.5 * critic_loss + actor_loss - beta * entropy;

            mOpt->zero_grad();
            loss.backward();
            mOpt->step();
        }
    }
};