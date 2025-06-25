#include "ppo.h"

using namespace std;


//#define DEBUG_TENSNORS
//#define LIMIT_ACTION_SPACE

void print_tensor_inline(const std::string& name, const torch::Tensor& t, int precision = 4, int max_elements = 10) {
	//std::cout << name << " shape: " << t.sizes() << std::endl;
#ifdef DEBUG_TENSNORS
	torch::Tensor flat = t.flatten().cpu();
	std::cout << name << "=tensor([";
	int64_t size = flat.size(0);
	std::cout << std::fixed << std::setprecision(precision);
	for (int64_t i = 0; i < std::min<int64_t>(size, max_elements / 2); ++i) {
	    std::cout << flat[i].item<double>() << ", ";
	}
	if (size > max_elements) {
	    std::cout << "...";
	    for (int64_t i = size - max_elements / 2; i < size; ++i) {
	        std::cout << ", " << flat[i].item<double>();
	    }
	}
	std::cout << "])" << std::endl << std::endl;
#endif
}
FeedForwardNNImpl::FeedForwardNNImpl(int in_dim, int out_dim, torch::Device& device) {
	try {
		layer1 = register_module("layer1", torch::nn::Linear(in_dim, 256));
		layer2 = register_module("layer2", torch::nn::Linear(256, 256));
		layer3 = register_module("layer3", torch::nn::Linear(256, 128));
		layer4 = register_module("layer4", torch::nn::Linear(128, out_dim));

		layer1->to(device);
		layer2->to(device);
		layer3->to(device);
		layer4->to(device);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in FeedForwardNNImpl constructor: " << e.what() << std::endl;
		throw;
	}
}

torch::Tensor FeedForwardNNImpl::forward(torch::Tensor obs) {
	try {
		auto activation1 = torch::relu(layer1(obs));
		auto activation2 = torch::relu(layer2(activation1));
		auto activation3 = torch::relu(layer3(activation2));
		return layer4(activation3);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in forward: " << e.what() << std::endl;
		throw;
	}
}

PPO::PPO(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, string actor_model, string critic_model)
	: env(env), device(device) {
	try {
		obs_dim = env.observation_space().shape[0];
		act_dim = env.action_space().shape[0];

		actor = FeedForwardNN(obs_dim, act_dim, device);
		critic = FeedForwardNN(obs_dim, 1, device);

		if (!actor_model.empty() && !critic_model.empty()) {
			cout << "Loading in " << actor_model << " and " << critic_model << "..." << endl;
			torch::load(actor, actor_model);
			torch::load(critic, critic_model);
			cout << "Successfully loaded." << endl;
		}
		else if (!actor_model.empty() || !critic_model.empty()) {
			cerr << "Error: Actor and Critic models must be Specified or Empty" << endl;
			exit(0);
		}
		else {
			cout << "Training from scratch." << endl;
		}

		_init_hyperparameters(hyperparameters);

		actor_optim = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(lr));
		critic_optim = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(lr));

		float variance = 0.5f;
		float std_value = std::sqrt(variance);
		std_dev = torch::full({ act_dim }, std_value).to(device);

		logger["delta_t"] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		logger["t_so_far"] = 0;
		logger["i_so_far"] = 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO constructor: " << e.what() << std::endl;
		throw;
	}
}

void PPO::learn(int total_timesteps) {
	try {
		std::cout << "Learning... Running " << max_timesteps_per_episode
			<< " timesteps per episode, " << timesteps_per_batch
			<< " timesteps per batch for a total of "
			<< total_timesteps << " timesteps" << std::endl;

		int t_so_far = 0;
		int i_so_far = 0;

		while (t_so_far < total_timesteps) {
			auto [batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths] = rollout_train();

			t_so_far += batch_lengths.sum().item<int>();
			i_so_far += 1;

			float frac = (t_so_far - 1.0f) / total_timesteps;
			float new_lr = lr * (1.0f - frac);
			new_lr = std::max(new_lr, 0.0f);
			actor_optim->param_groups()[0].options().set_lr(new_lr);
			critic_optim->param_groups()[0].options().set_lr(new_lr);

			logger["t_so_far"] = t_so_far;
			logger["i_so_far"] = i_so_far;

			auto [V, _] = evaluate(batch_obs, batch_acts);

			torch::Tensor A_k = batch_rtgs.unsqueeze(1) - V.detach();
			std::cout << "A_k shape: " << A_k.sizes() << std::endl;
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10);

			print_tensor_inline("batch_obs", batch_obs);
			print_tensor_inline("batch_acts", batch_acts);
			print_tensor_inline("batch_log_probs", batch_log_probs);
			print_tensor_inline("batch_rtgs", batch_rtgs);
			print_tensor_inline("batch_lengths", batch_lengths);
			print_tensor_inline("V", V);
			print_tensor_inline("A_k", A_k);

			for (int epoch = 0; epoch < n_updates_per_iteration; ++epoch) {
				auto [V, curr_log_probs] = evaluate(batch_obs, batch_acts);

				torch::Tensor ratios = torch::exp(curr_log_probs - batch_log_probs);
				std::cout << "ratios shape: " << ratios.sizes() << std::endl;
				torch::Tensor surr1 = ratios * A_k;
				torch::Tensor surr2 = torch::clamp(ratios, 1 - clip, 1 + clip) * A_k;

				torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
				torch::Tensor critic_loss = torch::mse_loss(V, batch_rtgs);


				print_tensor_inline("V", V);
				print_tensor_inline("curr_log_probs", curr_log_probs);
				print_tensor_inline("ratios", ratios);
				print_tensor_inline("surr1", surr1);
				print_tensor_inline("surr2", surr2);
				print_tensor_inline("actor_loss", actor_loss);
				print_tensor_inline("critic_loss", critic_loss);

				actor_optim->zero_grad();
				std::cout << "actor backward" << std::endl;
				actor_loss.backward({}, true);
				actor_optim->step();

				critic_optim->zero_grad();
				std::cout << "critic backward" << std::endl;
				critic_loss.backward();
				critic_optim->step();

				logger["actor_loss"] = actor_loss;

				print_tensor_inline("actor_loss backward", actor_loss);
				print_tensor_inline("critic_loss backward", critic_loss);
			}

			_log_train();

			if (i_so_far % save_freq == 0) {
				if (!std::filesystem::exists("./models")) {
					std::filesystem::create_directories("./models");
				}
				std::cout << "Saving training model as /models/ppo_actor.pt" << std::endl;
				torch::save(actor, "./models/ppo_actor.pt");
				torch::save(critic, "./models/ppo_critic.pt");
			}
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in learn: " << e.what() << std::endl;
		throw;
	}
}

void PPO::_init_hyperparameters(const unordered_map<string, float>& hyperparameters) {
	try {
		timesteps_per_batch = 4800;
		max_timesteps_per_episode = 1600;
		n_updates_per_iteration = 5;
		lr = 0.005;
		gamma = 0.95;
		clip = 0.2;
		render = false;
		render_every_i = 10;
		save_freq = 10;
		seed = nullopt;

		for (const auto& [param, val] : hyperparameters) {
			if (param == "timesteps_per_batch") timesteps_per_batch = static_cast<int>(val);
			else if (param == "max_timesteps_per_episode") max_timesteps_per_episode = static_cast<int>(val);
			else if (param == "n_updates_per_iteration") n_updates_per_iteration = static_cast<int>(val);
			else if (param == "lr") lr = val;
			else if (param == "gamma") gamma = val;
			else if (param == "clip") clip = val;
			else if (param == "render") render = (val != 0.0);
			else if (param == "render_every_i") render_every_i = static_cast<int>(val);
			else if (param == "save_freq") save_freq = static_cast<int>(val);
			else if (param == "seed") seed = static_cast<int>(val);
		}

		if (seed.has_value()) {
			torch::manual_seed(seed.value());
			cout << "Successfully set seed to " << seed.value() << endl;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in _init_hyperparameters: " << e.what() << std::endl;
		throw;
	}
}


void PPO::_log_train() {
	try {
		long long prev_delta_t = std::get<long long>(logger["delta_t"]);
		logger["delta_t"] = chrono::duration_cast<chrono::nanoseconds>(
			chrono::high_resolution_clock::now().time_since_epoch()).count();
		float delta_t_sec = (std::get<long long>(logger["delta_t"]) - prev_delta_t) / 1e9;

		stringstream delta_t_ss;
		delta_t_ss << fixed << setprecision(2) << delta_t_sec;
		string delta_t = delta_t_ss.str();

		int t_so_far = std::get<int>(logger["t_so_far"]);
		int i_so_far = std::get<int>(logger["i_so_far"]);

		vector<int> batch_lengths = std::get<vector<int>>(logger["batch_lengths"]);
		float avg_ep_lens = 0.0;
		if (!batch_lengths.empty()) {
			avg_ep_lens = accumulate(batch_lengths.begin(), batch_lengths.end(), 0.0) / batch_lengths.size();
		}

		vector<vector<float>> batch_rewards = std::get<vector<vector<float>>>(logger["batch_rewards"]);
		float avg_ep_rews = 0.0;
		if (!batch_rewards.empty()) {
			float sum_rews = 0.0;
			int count = 0;
			for (const auto& ep_rews : batch_rewards) {
				sum_rews += accumulate(ep_rews.begin(), ep_rews.end(), 0.0);
				count++;
			}
			avg_ep_rews = sum_rews / count;
		}

		auto actor_loss = std::get<torch::Tensor>(logger["actor_loss"]);
		float avg_actor_loss = 0.0;
		if (actor_loss.numel() > 0) {
			if (actor_loss.dim() == 0) {
				// scalar tensor
				avg_actor_loss = actor_loss.item<float>();
			}
			else {
				float sum_loss = 0.0;
				for (int i = 0; i < actor_loss.size(0); ++i) {
					sum_loss += actor_loss[i].item<float>();
				}
				avg_actor_loss = sum_loss / actor_loss.size(0);
			}
		}

		stringstream avg_ep_lens_ss, avg_ep_rews_ss, avg_actor_loss_ss;
		avg_ep_lens_ss << fixed << setprecision(2) << avg_ep_lens;
		avg_ep_rews_ss << fixed << setprecision(2) << avg_ep_rews;
		avg_actor_loss_ss << fixed << setprecision(5) << avg_actor_loss;

		cout << endl;
		cout << "-------------------- Iteration #" << i_so_far << " --------------------" << endl;
		cout << "Average Episodic Length: " << avg_ep_lens_ss.str() << endl;
		cout << "Average Episodic Return: " << avg_ep_rews_ss.str() << endl;
		cout << "Average Loss: " << avg_actor_loss_ss.str() << endl;
		cout << "Timesteps So Far: " << t_so_far << endl;
		cout << "Iteration took: " << delta_t << " secs" << endl;
		cout << "------------------------------------------------------" << endl;
		cout << endl;

		logger["batch_lengths"] = vector<int>{};
		logger["batch_rewards"] = vector<vector<float>>{};
		logger["actor_loss"] = torch::Tensor();
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in _log_summary: " << e.what() << std::endl;
		throw;
	}
}

torch::Tensor PPO::compute_rtgs(const vector<vector<float>>& batch_rewards) {
	try {
		vector<float> batch_rtgs;

		for (auto it = batch_rewards.rbegin(); it != batch_rewards.rend(); ++it) {
			const auto& ep_rews = *it;
			float discounted_reward = 0;
			for (auto rit = ep_rews.rbegin(); rit != ep_rews.rend(); ++rit) {
				discounted_reward = *rit + discounted_reward * gamma;
				batch_rtgs.insert(batch_rtgs.begin(), discounted_reward);
			}
		}

		return torch::tensor(batch_rtgs, torch::kFloat).to(device);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in compute_rtgs: " << e.what() << std::endl;
		throw;
	}
}

#ifdef  LIMIT_ACTION_SPACE
std::pair<torch::Tensor, torch::Tensor> PPO::get_action(const torch::Tensor& obs_tensor) {
	try {
		torch::Tensor mean = actor->forward(obs_tensor);
		auto dist = NormalMultivariate(mean, std_dev, device);
		torch::Tensor raw_action = dist.sample();
		torch::Tensor action_tensor = torch::tanh(raw_action);
		torch::Tensor log_prob = dist.log_prob(raw_action);
		torch::Tensor correction = torch::log(1 - action_tensor.pow(2) + 1e-6).sum(-1);
		log_prob = log_prob - correction;

		print_tensor_inline("obs_tensor", obs_tensor);
		print_tensor_inline("mean", mean);
		print_tensor_inline("raw_action", raw_action);
		print_tensor_inline("action_tensor", action_tensor);
		print_tensor_inline("corrected_log_prob", log_prob);

		return { action_tensor.detach(), log_prob.detach() };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO::get_action: " << e.what() << std::endl;
		throw;
	}
}

std::pair<torch::Tensor, torch::Tensor> PPO_Eval::get_action(const torch::Tensor& obs_tensor) {
	try {
		torch::Tensor mean = actor->forward(obs_tensor);
		auto dist = NormalMultivariate(mean, std_dev, device);
		torch::Tensor raw_action = dist.sample();
		torch::Tensor action_tensor = torch::tanh(raw_action);
		torch::Tensor log_prob = dist.log_prob(raw_action);
		torch::Tensor correction = torch::log(1 - action_tensor.pow(2) + 1e-6).sum(-1);
		log_prob = log_prob - correction;

		print_tensor_inline("obs_tensor", obs_tensor);
		print_tensor_inline("mean", mean);
		print_tensor_inline("raw_action", raw_action);
		print_tensor_inline("action_tensor", action_tensor);
		print_tensor_inline("corrected_log_prob", log_prob);

		return { action_tensor.detach(), log_prob.detach() };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO_Eval::get_action: " << e.what() << std::endl;
		throw;
	}
}
#else

std::pair<torch::Tensor, torch::Tensor> PPO::get_action(const torch::Tensor& obs_tensor) {
	try {
		torch::Tensor mean = actor->forward(obs_tensor);
		auto dist = NormalMultivariate(mean, std_dev, device);
		torch::Tensor action_tensor = dist.sample();
		torch::Tensor log_prob = dist.log_prob(action_tensor);

		print_tensor_inline("obs_tensor", obs_tensor);
		print_tensor_inline("mean", mean);
		print_tensor_inline("action_tensor", action_tensor);
		print_tensor_inline("log_prob_tensor", log_prob);

		return { action_tensor.detach(), log_prob.detach() };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO::get_action: " << e.what() << std::endl;
		throw;
	}
}

std::pair<torch::Tensor, torch::Tensor> PPO_Eval::get_action(const torch::Tensor& obs_tensor) {
	try {
		torch::Tensor mean = actor->forward(obs_tensor);
		auto dist = NormalMultivariate(mean, std_dev, device);
		torch::Tensor action_tensor = dist.sample();
		torch::Tensor log_prob = dist.log_prob(action_tensor);

		print_tensor_inline("obs_tensor", obs_tensor);
		print_tensor_inline("mean", mean);
		print_tensor_inline("action_tensor", action_tensor);
		print_tensor_inline("log_prob_tensor", log_prob);

		return { action_tensor.detach(), log_prob.detach() };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO_Eval::get_action: " << e.what() << std::endl;
		throw;
	}
}
#endif //  LIMIT_ACTION_SPACE
pair<torch::Tensor, torch::Tensor> PPO::evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts) {
	try {
		torch::Tensor V = critic->forward(batch_obs);
		print_tensor_inline("V", V);
		torch::Tensor mean = actor->forward(batch_obs);
		print_tensor_inline("mean", mean);
		NormalMultivariate dist(mean, std_dev, device);
		torch::Tensor log_probs = dist.log_prob(batch_acts);
		print_tensor_inline("log_probs", log_probs);
		return { V, log_probs };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in evaluate: " << e.what() << std::endl;
		throw;
	}
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PPO::rollout_train() {
	try {
		vector<torch::Tensor> batch_obs_vec;
		vector<torch::Tensor> batch_acts_vec;
		vector<torch::Tensor> batch_log_probs_vec;
		vector<vector<float>> batch_rewards;
		vector<int> batch_lengths_vec;

		vector<float> ep_rews;
		int t = 0;

		while (t < timesteps_per_batch) {
			ep_rews.clear();
			auto [obs_tensor, _] = env.reset();
			bool done = false;

			for (int ep_t = 0; ep_t < max_timesteps_per_episode; ++ep_t) {
				if (render && (std::get<int>(logger["i_so_far"]) % render_every_i == 0) && batch_lengths_vec.empty()) {
					env.render();
				}

				t += 1;
				batch_obs_vec.push_back(obs_tensor);

				auto [action_tensor, log_prob] = get_action(obs_tensor);
				auto [next_obs, rew, terminated, truncated, __] = env.step(action_tensor);
				print_tensor_inline("log_prob", log_prob);

				done = terminated || truncated;

				ep_rews.push_back(rew);
				batch_acts_vec.push_back(action_tensor);
				batch_log_probs_vec.push_back(log_prob);

				obs_tensor = next_obs;

				if (done) {
					break;
				}
			}

			batch_lengths_vec.push_back(ep_rews.size());
			batch_rewards.push_back(ep_rews);
		}

		torch::Tensor batch_obs = torch::stack(batch_obs_vec).to(torch::kFloat);
		torch::Tensor batch_acts = torch::stack(batch_acts_vec).to(torch::kFloat);
		torch::Tensor batch_log_probs = torch::stack(batch_log_probs_vec).to(torch::kFloat);
		torch::Tensor batch_rtgs = compute_rtgs(batch_rewards);
		torch::Tensor batch_lengths = torch::tensor(batch_lengths_vec, torch::kInt64);

		print_tensor_inline("batch_obs", batch_obs);
		print_tensor_inline("batch_acts", batch_acts);
		print_tensor_inline("batch_log_probs", batch_log_probs);
		print_tensor_inline("batch_rtgs", batch_rtgs);
		print_tensor_inline("batch_lengths", batch_lengths);
		logger["batch_rewards"] = batch_rewards;
		logger["batch_lengths"] = batch_lengths_vec;

		return { batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths };
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in rollout_train: " << e.what() << std::endl;
		throw;
	}
}

PPO_Eval::PPO_Eval(Env& env, torch::Device& device, string actor_model)
	: env(env), device(device) {
	try {
		if (actor_model.empty()) {
			cerr << "No actor model file. Exiting." << endl;
			exit(0);
		}

		obs_dim = env.observation_space().shape[0];
		act_dim = env.action_space().shape[0];

		actor = FeedForwardNN(obs_dim, act_dim, device);
		torch::load(actor, actor_model);

		float variance = 0.5f;
		float std_value = std::sqrt(variance);
		std_dev = torch::full({ act_dim }, std_value).to(device);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in PPO_Eval constructor: " << e.what() << std::endl;
		throw;
	}
}

void PPO_Eval::eval_policy(bool render, float fixedTimeStepS) {
	try {
		int ep_num = 0;

		while (true) {
			auto [obs_tensor, _] = env.reset();
			bool done = false;

			int t = 0;
			float ep_len = 0.0f;
			float ep_ret = 0.0f;

			while (!done) {
				t++;

				if (render) {
					env.render();
				}
				auto [action_tensor, log_prob] = get_action(obs_tensor);
				auto [next_obs, rew, terminated, truncated, __] = env.step(action_tensor);
				done = terminated || truncated;

				ep_ret += rew;
				obs_tensor = next_obs;
				if (fixedTimeStepS > 0.001f)
					b3Clock::usleep(1000. * 1000. * fixedTimeStepS);
			}

			ep_len = static_cast<float>(t);
			log_eval(ep_len, ep_ret, ep_num);
			ep_num++;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in eval_policy: " << e.what() << std::endl;
		throw;
	}
}


void PPO_Eval::log_eval(float ep_len, float ep_ret, int ep_num) {
	// Round decimals for nicer output
	ep_len = std::round(ep_len * 100.0f) / 100.0f;
	ep_ret = std::round(ep_ret * 100.0f) / 100.0f;

	std::cout << std::endl;
	std::cout << "-------------------- Episode #" << ep_num << " --------------------" << std::endl;
	std::cout << "Episodic Length: " << ep_len << std::endl;
	std::cout << "Episodic Return: " << ep_ret << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << std::endl;
	std::cout.flush();
}
