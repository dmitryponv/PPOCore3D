#include "ppo.h"

using namespace std;


//#define DEBUG_TENSNORS
//#define LIMIT_ACTION_SPACE

void print_tensor_inline(const std::string& name, const torch::Tensor& t, int precision = 4, int max_elements = 10) {
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


MultivariateNormal::MultivariateNormal(const torch::Tensor& loc,
	const torch::optional<torch::Tensor>& covariance_matrix,
	const torch::optional<torch::Tensor>& precision_matrix,
	const torch::optional<torch::Tensor>& scale_tril) {
	if (loc.dim() < 1) {
		throw std::invalid_argument("loc must be at least one-dimensional.");
	}

	int specified = (bool)covariance_matrix + (bool)precision_matrix + (bool)scale_tril;
	if (specified != 1) {
		throw std::invalid_argument("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified.");
	}

	if (scale_tril.has_value()) {
		if (scale_tril->dim() < 2) {
			throw std::invalid_argument("scale_tril must be at least two-dimensional.");
		}
		auto bshape = broadcast_shapes(scale_tril->sizes().vec(), loc.sizes().vec(), 2, 1);
		scale_tril_ = scale_tril->expand(bshape).contiguous();
		_unbroadcasted_scale_tril = *scale_tril;
		batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
	}
	else if (covariance_matrix.has_value()) {
		if (covariance_matrix->dim() < 2) {
			throw std::invalid_argument("covariance_matrix must be at least two-dimensional.");
		}
		auto bshape = broadcast_shapes(covariance_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
		covariance_matrix_ = covariance_matrix->expand(bshape).contiguous();
		_unbroadcasted_scale_tril = torch::linalg_cholesky(*covariance_matrix);
		batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
	}
	else {
		if (precision_matrix->dim() < 2) {
			throw std::invalid_argument("precision_matrix must be at least two-dimensional.");
		}
		auto bshape = broadcast_shapes(precision_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
		precision_matrix_ = precision_matrix->expand(bshape).contiguous();
		_unbroadcasted_scale_tril = torch::linalg_cholesky(torch::linalg_inv(*precision_matrix));
		batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
	}

	auto expanded_shape = batch_shape;
	expanded_shape.push_back(loc.size(-1));
	this->loc = loc.expand(expanded_shape).contiguous();
	event_shape = { loc.size(-1) };
}

torch::Tensor MultivariateNormal::scale_tril() const {
	auto shape = batch_shape;
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	return _unbroadcasted_scale_tril.expand(shape);
}

torch::Tensor MultivariateNormal::covariance_matrix() const {
	auto L = _unbroadcasted_scale_tril;
	auto shape = batch_shape;
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	return torch::matmul(L, L.transpose(-2, -1)).expand(shape);
}

torch::Tensor MultivariateNormal::precision_matrix() const {
	auto shape = batch_shape;
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	return torch::cholesky_inverse(_unbroadcasted_scale_tril).expand(shape);
}

torch::Tensor MultivariateNormal::mean() const {
	return loc;
}

torch::Tensor MultivariateNormal::mode() const {
	return loc;
}

torch::Tensor MultivariateNormal::variance() const {
	auto shape = batch_shape;
	shape.insert(shape.end(), event_shape.begin(), event_shape.end());
	return _unbroadcasted_scale_tril.pow(2).sum(-1).expand(shape);
}

torch::Tensor MultivariateNormal::sample(const std::vector<int64_t>& sample_shape) const {
	torch::NoGradGuard no_grad;
	auto shape = sample_shape;
	shape.insert(shape.end(), loc.sizes().begin(), loc.sizes().end());
	auto eps = torch::randn(shape, loc.options());
	auto L = _unbroadcasted_scale_tril;
	auto result = loc + torch::matmul(L, eps.unsqueeze(-1)).squeeze(-1);
	return result;
}

torch::Tensor MultivariateNormal::log_prob(const torch::Tensor& value) const {
	auto diff = value.to(loc.device()) - loc;
	auto M = batch_mahalanobis(_unbroadcasted_scale_tril, diff);
	auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
	return -0.5 * (event_shape[0] * std::log(2 * M_PI) + M) - half_log_det;
}

torch::Tensor MultivariateNormal::entropy() const {
	auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
	return 0.5 * event_shape[0] * (1.0 + std::log(2 * M_PI)) + half_log_det;
}

torch::Tensor MultivariateNormal::batch_mahalanobis(const torch::Tensor& L, const torch::Tensor& diff) {
	auto solve = torch::linalg_solve_triangular(L, diff.unsqueeze(-1), /*upper=*/false).squeeze(-1);
	return solve.pow(2).sum(-1);
}

std::vector<int64_t> MultivariateNormal::broadcast_shapes(std::vector<int64_t> a, std::vector<int64_t> b, int a_end, int b_end) {
	auto a_prefix = std::vector<int64_t>(a.begin(), a.end() - a_end);
	auto b_prefix = std::vector<int64_t>(b.begin(), b.end() - b_end);
	size_t ndim = std::max(a_prefix.size(), b_prefix.size());
	std::vector<int64_t> result(ndim, 1);
	for (int i = ndim - 1, ai = a_prefix.size() - 1, bi = b_prefix.size() - 1; i >= 0; --i, --ai, --bi) {
		int64_t a_dim = ai >= 0 ? a_prefix[ai] : 1;
		int64_t b_dim = bi >= 0 ? b_prefix[bi] : 1;
		if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
			throw std::invalid_argument("Incompatible shapes for broadcasting");
		}
		result[i] = std::max(a_dim, b_dim);
	}
	result.insert(result.end(), a.end() - a_end, a.end());
	return result;
}

FeedForwardNNImpl::FeedForwardNNImpl(int in_dim, int out_dim, torch::Device& device) {

	layer1 = register_module("layer1", torch::nn::Linear(in_dim, 64));
	layer2 = register_module("layer2", torch::nn::Linear(64, 64));
	layer3 = register_module("layer3", torch::nn::Linear(64, out_dim));

	layer1->to(device);
	layer2->to(device);
	layer3->to(device);
}

torch::Tensor FeedForwardNNImpl::forward(torch::Tensor obs) {
	try {
		//obs = obs.to(layer1->weight.device());
		auto activation1 = torch::relu(layer1(obs));
		auto activation2 = torch::relu(layer2(activation1));
		return layer3(activation2);
	}
	catch (const std::exception& e) {
		std::cerr << "Exception in forward: " << e.what() << std::endl;
		throw;  // rethrow or handle as needed
	}
}

PPO::PPO(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, string actor_model, string critic_model)
	: env(env), device(device) {

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

	cov_var = torch::full({ act_dim }, 0.5).to(device);
	cov_mat = cov_var.diag();

	logger["delta_t"] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	logger["t_so_far"] = 0;
	logger["i_so_far"] = 0;
}

void PPO::learn(int total_timesteps) {
	std::cout << "Learning... Running " << max_timesteps_per_episode
		<< " timesteps per episode, " << timesteps_per_batch
		<< " timesteps per batch for a total of "
		<< total_timesteps << " timesteps" << std::endl;

	int t_so_far = 0;
	int i_so_far = 0;

	while (t_so_far < total_timesteps) {
		// ALG STEP 2-3: Rollout to collect a batch of trajectories
		auto [batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths] = rollout_train();

		// Count how many timesteps we collected
		t_so_far += batch_lengths.sum().item<int>();

		// Count iterations
		i_so_far += 1;


		//Update learning rate 
		float frac = (t_so_far - 1.0f) / total_timesteps;
		float new_lr = lr * (1.0f - frac);
		new_lr = std::max(new_lr, 0.0f);
		actor_optim->param_groups()[0].options().set_lr(new_lr);
		critic_optim->param_groups()[0].options().set_lr(new_lr);

		// Logging
		logger["t_so_far"] = t_so_far;
		logger["i_so_far"] = i_so_far;

		// Evaluate current value function and policy log probs
		auto [V, _] = evaluate(batch_obs, batch_acts);

		// Advantage estimation
		torch::Tensor A_k = batch_rtgs - V.detach();

		// Normalize advantages
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10);

		///TEMP
		print_tensor_inline("batch_obs", batch_obs);
		print_tensor_inline("batch_acts", batch_acts);
		print_tensor_inline("batch_log_probs", batch_log_probs);
		print_tensor_inline("batch_rtgs", batch_rtgs);
		print_tensor_inline("batch_lengths", batch_lengths);
		print_tensor_inline("V", V);
		print_tensor_inline("A_k", A_k);

		// PPO update for multiple epochs
		for (int epoch = 0; epoch < n_updates_per_iteration; ++epoch) {
			auto [V, curr_log_probs] = evaluate(batch_obs, batch_acts);

			// Compute ratio of new and old policy probabilities
			torch::Tensor ratios = torch::exp(curr_log_probs - batch_log_probs);

			// Compute surrogate losses
			torch::Tensor surr1 = ratios * A_k;
			torch::Tensor surr2 = torch::clamp(ratios, 1 - clip, 1 + clip) * A_k;

			// Compute losses
			torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
			torch::Tensor critic_loss = torch::mse_loss(V, batch_rtgs);

			// Backpropagate actor loss
			actor_optim->zero_grad();
			actor_loss.backward({}, /* retain_graph */ true);
			actor_optim->step();

			// Backpropagate critic loss
			critic_optim->zero_grad();
			critic_loss.backward();
			critic_optim->step();

			// Logging actor loss
			logger["actor_loss"] = actor_loss;

			///TEMP
			print_tensor_inline("V", V);
			print_tensor_inline("curr_log_probs", curr_log_probs);
			print_tensor_inline("ratios", ratios);
			print_tensor_inline("surr1", surr1);
			print_tensor_inline("surr2", surr2);
			print_tensor_inline("actor_loss", actor_loss);
			print_tensor_inline("critic_loss", critic_loss);

		}
		// Print training summary
		_log_train();

		// Save model
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

void PPO::_init_hyperparameters(const unordered_map<string, float>& hyperparameters) {
	// Initialize default values for hyperparameters
	timesteps_per_batch = 4800;               // Number of timesteps to run per batch
	max_timesteps_per_episode = 1600;         // Max number of timesteps per episode
	n_updates_per_iteration = 5;              // Number of times to update actor/critic per iteration
	lr = 0.005;                              // Learning rate of actor optimizer
	gamma = 0.95;                            // Discount factor to be applied when calculating Rewards-To-Go
	clip = 0.2;                              // Recommended 0.2, helps define the threshold to clip the ratio during SGA

	// Miscellaneous parameters
	render = false;                          // If we should render during rollout
	render_every_i = 10;                    // Only render every n iterations
	save_freq = 10;                         // How often we save in number of iterations
	seed = nullopt;                    // Sets the seed of our program, used for reproducibility of results

	// Change any default values to custom values for specified hyperparameters
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
		// Add more parameters here if needed
	}

	// Sets the seed if specified
	if (seed.has_value()) {
		// Set the seed
		torch::manual_seed(seed.value());
		cout << "Successfully set seed to " << seed.value() << endl;
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
	}
}

torch::Tensor PPO::compute_rtgs(const vector<vector<float>>& batch_rewards) {
	// The rewards-to-go (rtg) per episode per batch to return.
	// The shape will be (num timesteps per episode)
	vector<float> batch_rtgs;

	// Iterate through each episode
	for (auto it = batch_rewards.rbegin(); it != batch_rewards.rend(); ++it) {
		const auto& ep_rews = *it;
		float discounted_reward = 0; // The discounted reward so far

		// Iterate through all rewards in the episode. We go backwards for smoother calculation of each
		// discounted return (think about why it would be harder starting from the beginning)
		for (auto rit = ep_rews.rbegin(); rit != ep_rews.rend(); ++rit) {
			discounted_reward = *rit + discounted_reward * gamma;
			batch_rtgs.insert(batch_rtgs.begin(), discounted_reward);
		}
	}

	// Convert the rewards-to-go into a tensor
	return torch::tensor(batch_rtgs, torch::kFloat).to(device);
}

#ifdef  LIMIT_ACTION_SPACE
std::pair<torch::Tensor, torch::Tensor> PPO::get_action(const torch::Tensor& obs_tensor) {
	// Query the actor network for a mean action
	torch::Tensor mean = actor->forward(obs_tensor);

	// Create a distribution with the mean and covariance
	auto dist = MultivariateNormal(mean, cov_mat);

	// Sample an action (pre-squash)
	torch::Tensor raw_action = dist.sample();

	// Squash the action to [-1, 1] range
	torch::Tensor action_tensor = torch::tanh(raw_action);

	// Log prob of the unsquashed action
	torch::Tensor log_prob = dist.log_prob(raw_action);

	// Tanh correction: subtract log(det(Jacobian)) of the transformation
	// For tanh: log(1 - tanh(x)^2) = log(1 - action^2)
	torch::Tensor correction = torch::log(1 - action_tensor.pow(2) + 1e-6).sum(-1);

	// Adjusted log_prob
	log_prob = log_prob - correction;

	// Optional: print tensors
	print_tensor_inline("obs_tensor", obs_tensor);
	print_tensor_inline("mean", mean);
	print_tensor_inline("raw_action", raw_action);
	print_tensor_inline("action_tensor", action_tensor);
	print_tensor_inline("corrected_log_prob", log_prob);

	return { action_tensor, log_prob.detach() };
}
std::pair<torch::Tensor, torch::Tensor> PPO_Eval::get_action(const torch::Tensor& obs_tensor) {
	// Query the actor network for a mean action
	torch::Tensor mean = actor->forward(obs_tensor);

	// Create a distribution with the mean and covariance
	auto dist = MultivariateNormal(mean, cov_mat);

	// Sample an action (pre-squash)
	torch::Tensor raw_action = dist.sample();

	// Squash the action to [-1, 1] range
	torch::Tensor action_tensor = torch::tanh(raw_action);

	// Log prob of the unsquashed action
	torch::Tensor log_prob = dist.log_prob(raw_action);

	// Tanh correction: subtract log(det(Jacobian)) of the transformation
	// For tanh: log(1 - tanh(x)^2) = log(1 - action^2)
	torch::Tensor correction = torch::log(1 - action_tensor.pow(2) + 1e-6).sum(-1);

	// Adjusted log_prob
	log_prob = log_prob - correction;

	// Optional: print tensors
	print_tensor_inline("obs_tensor", obs_tensor);
	print_tensor_inline("mean", mean);
	print_tensor_inline("raw_action", raw_action);
	print_tensor_inline("action_tensor", action_tensor);
	print_tensor_inline("corrected_log_prob", log_prob);

	return { action_tensor, log_prob.detach() };
}
#else
std::pair<torch::Tensor, torch::Tensor> PPO::get_action(const torch::Tensor& obs_tensor) {
	// Query the actor network for a mean action
	torch::Tensor mean = actor->forward(obs_tensor);

	// Create a distribution with the mean action and std from the covariance matrix
	auto dist = MultivariateNormal(mean, cov_mat);

	// Sample an action from the distribution
	torch::Tensor action_tensor = dist.sample();// torch::tanh(dist.sample());

	// Compute the log probability with correction for tanh
	torch::Tensor log_prob = dist.log_prob(action_tensor);

	print_tensor_inline("obs_tensor", obs_tensor);
	print_tensor_inline("mean", mean);
	print_tensor_inline("action_tensor", action_tensor);
	print_tensor_inline("log_prob_tensor", log_prob);

	return { action_tensor, log_prob.detach() };
}
std::pair<torch::Tensor, torch::Tensor> PPO_Eval::get_action(const torch::Tensor& obs_tensor) {
	// Query the actor network for a mean action
	torch::Tensor mean = actor->forward(obs_tensor);

	// Create a distribution with the mean action and std from the covariance matrix
	auto dist = MultivariateNormal(mean, cov_mat);

	// Sample an action from the distribution
	torch::Tensor action_tensor = dist.sample();// torch::tanh(dist.sample());

	// Compute the log probability with correction for tanh
	torch::Tensor log_prob = dist.log_prob(action_tensor);

	print_tensor_inline("obs_tensor", obs_tensor);
	print_tensor_inline("mean", mean);
	print_tensor_inline("action_tensor", action_tensor);
	print_tensor_inline("log_prob_tensor", log_prob);

	return { action_tensor, log_prob.detach() };
}
#endif //  LIMIT_ACTION_SPACE





pair<torch::Tensor, torch::Tensor> PPO::evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts) {
	//Estimate the values of each observation, and the log probs of
	//each action in the most recent batch with the most recent
	//iteration of the actor network. Should be called from learn.

	// Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
	torch::Tensor V = critic->forward(batch_obs).squeeze();

	// Calculate the log probabilities of batch actions using most recent actor network.
	// This segment of code is similar to that in get_action()
	torch::Tensor mean = actor->forward(batch_obs);
	MultivariateNormal dist(mean, cov_mat);
	torch::Tensor log_probs = (dist.log_prob(batch_acts));

	// Return the value vector V of each observation in the batch
	// and log probabilities log_probs of each action in the batch
	return { V, log_probs };
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PPO::rollout_train() {
	// Batch data. For more details, check function header.
	vector<torch::Tensor> batch_obs_vec;
	vector<torch::Tensor> batch_acts_vec;
	vector<torch::Tensor> batch_log_probs_vec;
	vector<vector<float>> batch_rewards;
	vector<int> batch_lengths_vec;

	// Episodic data. Keeps track of rewards per episode, will get cleared
	// upon each new episode
	vector<float> ep_rews;

	int t = 0; // Keeps track of how many timesteps we've run so far this batch

	// Keep simulating until we've run more than or equal to specified timesteps per batch
	while (t < timesteps_per_batch) {
		ep_rews.clear(); // rewards collected per episode

		// Reset the environment. Note that obs is short for observation.
		auto [obs_tensor, _] = env.reset();
		bool done = false;

		// Run an episode for a maximum of max_timesteps_per_episode timesteps
		for (int ep_t = 0; ep_t < max_timesteps_per_episode; ++ep_t) {
			// If render is specified, render the environment
			if (render && (std::get<int>(logger["i_so_far"]) % render_every_i == 0) && batch_lengths_vec.empty()) {
				env.render();
			}

			t += 1; // Increment timesteps ran this batch so far

			// Track observations in this batch
			batch_obs_vec.push_back(obs_tensor);

			// Calculate action and make a step in the env.
			// Note that rew is short for reward.
			auto [action_tensor, log_prob] = get_action(obs_tensor);
			auto [next_obs, rew, terminated, truncated, __] = env.step(action_tensor);
			print_tensor_inline("log_prob", log_prob);

			// Don't really care about the difference between terminated or truncated in this, so just combine them
			done = terminated || truncated;

			// Track recent reward, action, and action log probability
			ep_rews.push_back(rew);
			batch_acts_vec.push_back(action_tensor);
			batch_log_probs_vec.push_back(log_prob);

			obs_tensor = next_obs;

			// If the environment tells us the episode is terminated, break
			if (done) {
				break;
			}
		}

		// Track episodic lengths and rewards
		batch_lengths_vec.push_back(ep_rews.size());
		batch_rewards.push_back(ep_rews);
	}

	// Reshape data as tensors in the shape specified in function description, before returning
	torch::Tensor batch_obs = torch::stack(batch_obs_vec).to(torch::kFloat);
	torch::Tensor batch_acts = torch::stack(batch_acts_vec).to(torch::kFloat);
	torch::Tensor batch_log_probs = torch::stack(batch_log_probs_vec).to(torch::kFloat);
	torch::Tensor batch_rtgs = compute_rtgs(batch_rewards); // ALG STEP 4
	torch::Tensor batch_lengths = torch::tensor(batch_lengths_vec, torch::kInt64);

	print_tensor_inline("batch_obs", batch_obs);
	print_tensor_inline("batch_acts", batch_acts);
	print_tensor_inline("batch_log_probs", batch_log_probs);
	print_tensor_inline("batch_rtgs", batch_rtgs);
	print_tensor_inline("batch_lengths", batch_lengths);

	// Log the episodic returns and episodic lengths in this batch.
	logger["batch_rewards"] = batch_rewards;
	logger["batch_lengths"] = batch_lengths_vec;

	return { batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths };
}

PPO_Eval::PPO_Eval(Env& env, torch::Device& device, string actor_model)
	: env(env), device(device) {

	if (actor_model.empty()) {
		cerr << "No actor model file. Exiting." << endl;
		exit(0);
	}


	obs_dim = env.observation_space().shape[0];
	act_dim = env.action_space().shape[0];

	actor = FeedForwardNN(obs_dim, act_dim, device);

	torch::load(actor, actor_model);

	cov_mat = torch::full({ act_dim }, 0.5).to(device).diag();
}

void PPO_Eval::eval_policy(bool render, float fixedTimeStepS) {
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
			if(fixedTimeStepS > 0.001f)
				b3Clock::usleep(1000. * 1000. * fixedTimeStepS);
		}

		ep_len = static_cast<float>(t);

		log_eval(ep_len, ep_ret, ep_num);
		ep_num++;
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
