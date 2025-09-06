#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <torch/torch.h>
#include <numeric>

class NormalMultivariate {
private:
    torch::Tensor mean_;
    torch::Tensor scale_tril_;
    torch::Device& device_;

    // Manual implementation of Cholesky decomposition for compatibility with older LibTorch versions.
    // This function assumes the input matrix is symmetric and positive-definite.
    static torch::Tensor cholesky_decomposition(const torch::Tensor& A) {
        if (A.dim() != 2 || A.size(0) != A.size(1)) {
            throw std::invalid_argument("Input must be a square matrix for Cholesky decomposition.");
        }
        int64_t n = A.size(0);
        auto L = torch::zeros_like(A);

        auto A_acc = A.accessor<float, 2>();
        auto L_acc = L.accessor<float, 2>();

        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                float sum = 0.0;
                if (j == i) {
                    // Diagonal elements
                    for (int64_t k = 0; k < j; ++k) {
                        sum += L_acc[i][k] * L_acc[i][k];
                    }
                    L_acc[i][i] = std::sqrt(A_acc[i][i] - sum);
                }
                else {
                    // Non-diagonal elements
                    for (int64_t k = 0; k < j; ++k) {
                        sum += L_acc[i][k] * L_acc[j][k];
                    }
                    L_acc[i][j] = (A_acc[i][j] - sum) / L_acc[j][j];
                }
            }
        }
        return L;
    }

public:
    NormalMultivariate(const torch::Tensor& mean, const torch::Tensor& cov_mat, torch::Device& device)
        : mean_(mean), device_(device) {
        // Calculate the Cholesky decomposition using the manual implementation
        scale_tril_ = NormalMultivariate::cholesky_decomposition(cov_mat);
    }

    torch::Tensor sample() {
        // Sample from a standard normal distribution
        auto normal_noise = torch::randn(mean_.sizes(), mean_.options());
        // Transform the sample using the Cholesky factor and add the mean
        return mean_ + torch::matmul(scale_tril_, normal_noise.unsqueeze(-1)).squeeze(-1);
    }

    torch::Tensor log_prob(const torch::Tensor& value) {
        // Calculate the difference from the mean
        auto diff = value - mean_;

        // Calculate the Mahalanobis distance term
        // This is equivalent to `||inv(L) * diff||^2`
        auto M_term = std::get<0>(torch::triangular_solve(diff.unsqueeze(-1), scale_tril_, false)).pow(2).sum(-1);

        // Calculate the log determinant term
        auto half_log_det = scale_tril_.diagonal(0, -2, -1).log().sum(-1);

        // Combine the terms to get the log probability
        auto log_prob_tensor = -0.5 * (diff.size(-1) * std::log(2 * M_PI) + M_term) - half_log_det;

        return log_prob_tensor;
    }
};







//namespace {
//    // A helper function for batched matrix-vector product.
//    torch::Tensor _batch_mv(const torch::Tensor& bmat, const torch::Tensor& bvec) {
//        return torch::matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1);
//    }
//
//    // A helper to compute the squared Mahalanobis distance for a factored M = L*L.T.
//    // Handles batches for both bL and bx.
//    torch::Tensor _batch_mahalanobis(const torch::Tensor& bL, const torch::Tensor& bx) {
//        int64_t n = bx.size(-1);
//        auto bx_batch_shape = bx.sizes().vec();
//        bx_batch_shape.pop_back();
//
//        int64_t bx_batch_dims = bx_batch_shape.size();
//        int64_t bL_batch_dims = bL.dim() - 2;
//        int64_t outer_batch_dims = bx_batch_dims - bL_batch_dims;
//        int64_t old_batch_dims = outer_batch_dims + bL_batch_dims;
//        int64_t new_batch_dims = outer_batch_dims + 2 * bL_batch_dims;
//
//        // Reshape bx with the shape (..., 1, i, j, 1, n)
//        std::vector<int64_t> bx_new_shape(bx.sizes().vec());
//        bx_new_shape.resize(outer_batch_dims);
//        for (int64_t i = 0; i < bL_batch_dims; ++i) {
//            int64_t sx = bx.sizes()[outer_batch_dims + i];
//            int64_t sL = bL.sizes()[i];
//            bx_new_shape.push_back(sx / sL);
//            bx_new_shape.push_back(sL);
//        }
//        bx_new_shape.push_back(n);
//        auto bx_reshaped = bx.reshape(bx_new_shape);
//
//        // Permute bx to make it have shape (..., 1, j, i, 1, n)
//        std::vector<int64_t> permute_dims;
//        permute_dims.reserve(new_batch_dims + 1);
//        for (int64_t i = 0; i < outer_batch_dims; ++i) {
//            permute_dims.push_back(i);
//        }
//        for (int64_t i = 0; i < bL_batch_dims; ++i) {
//            permute_dims.push_back(outer_batch_dims + 2 * i);
//        }
//        for (int64_t i = 0; i < bL_batch_dims; ++i) {
//            permute_dims.push_back(outer_batch_dims + 2 * i + 1);
//        }
//        permute_dims.push_back(new_batch_dims);
//        auto bx_permuted = bx_reshaped.permute(permute_dims);
//
//        // Perform the solve and sum
//        auto flat_L = bL.reshape({ -1, n, n });
//        auto flat_x = bx_permuted.reshape({ -1, flat_L.size(0), n });
//        auto flat_x_swap = flat_x.permute({ 1, 2, 0 });
//        auto M_swap = std::get<0>(torch::triangular_solve(flat_x_swap, flat_L, false)).pow(2).sum(-2);
//        auto M = M_swap.t();
//
//        // Now we revert the above reshape and permute operators.
//        auto permuted_M = M.reshape(bx_permuted.sizes().vec());
//
//        std::vector<int64_t> permute_inv_dims;
//        permute_inv_dims.reserve(old_batch_dims + bL_batch_dims);
//        for (int64_t i = 0; i < outer_batch_dims; ++i) {
//            permute_inv_dims.push_back(i);
//        }
//        for (int64_t i = 0; i < bL_batch_dims; ++i) {
//            permute_inv_dims.push_back(outer_batch_dims + i);
//            permute_inv_dims.push_back(old_batch_dims + i);
//        }
//        auto reshaped_M = permuted_M.permute(permute_inv_dims);
//
//        return reshaped_M.reshape(bx_batch_shape);
//    }
//
//    // A helper to compute scale_tril from precision matrix
//    torch::Tensor _precision_to_scale_tril(const torch::Tensor& P) {
//        auto Lf = torch::cholesky(torch::flip(P, { -2, -1 }));
//        auto L_inv = Lf.transpose(-2, -1).flip({ -2, -1 });
//        auto Id = torch::eye(P.size(-1), P.options());
//        return std::get<0>(torch::triangular_solve(Id, L_inv, false));
//    }
//
//    // A helper to compute the broadcasted shape
//    std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
//        if (shape1.size() > shape2.size()) {
//            return broadcast_shapes(shape2, shape1);
//        }
//        std::vector<int64_t> result(shape2.size());
//        int64_t offset = shape2.size() - shape1.size();
//        for (size_t i = 0; i < shape1.size(); ++i) {
//            if (shape1[i] != 1 && shape2[offset + i] != 1 && shape1[i] != shape2[offset + i]) {
//                throw std::runtime_error("The expanded size of the tensor must match the existing size at each dimension.");
//            }
//            result[offset + i] = std::max(shape1[i], shape2[offset + i]);
//        }
//        for (size_t i = 0; i < offset; ++i) {
//            result[i] = shape2[i];
//        }
//        return result;
//    }
//}
//class NormalMultivariate2 {
//private:
//    torch::Tensor loc_;
//    torch::Tensor _unbroadcasted_scale_tril;
//    std::vector<int64_t> _batch_shape;
//    std::vector<int64_t> _event_shape;
//
//public:
//    NormalMultivariate(
//        const torch::Tensor& loc,
//        const c10::optional<torch::Tensor>& covariance_matrix = c10::nullopt,
//        const c10::optional<torch::Tensor>& precision_matrix = c10::nullopt,
//        const c10::optional<torch::Tensor>& scale_tril = c10::nullopt
//    ) : loc_(loc) {
//
//        if (loc.dim() < 1) {
//            throw std::invalid_argument("loc must be at least one-dimensional.");
//        }
//
//        int count = (covariance_matrix.has_value() ? 1 : 0) +
//            (precision_matrix.has_value() ? 1 : 0) +
//            (scale_tril.has_value() ? 1 : 0);
//
//        if (count != 1) {
//            throw std::invalid_argument("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.");
//        }
//
//        int64_t event_dim = loc.size(-1);
//        auto loc_batch_shape_vec = loc.sizes().vec();
//        loc_batch_shape_vec.pop_back();
//
//        if (covariance_matrix.has_value()) {
//            _unbroadcasted_scale_tril = torch::cholesky(covariance_matrix.value());
//            _batch_shape = broadcast_shapes(_unbroadcasted_scale_tril.sizes().vec(), loc_batch_shape_vec);
//        }
//        else if (precision_matrix.has_value()) {
//            _unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix.value());
//            _batch_shape = broadcast_shapes(_unbroadcasted_scale_tril.sizes().vec(), loc_batch_shape_vec);
//        }
//        else if (scale_tril.has_value()) {
//            _unbroadcasted_scale_tril = scale_tril.value();
//            _batch_shape = broadcast_shapes(_unbroadcasted_scale_tril.sizes().vec(), loc_batch_shape_vec);
//        }
//
//        _event_shape = { event_dim };
//    }
//
//    torch::Tensor sample(const std::vector<int64_t>& sample_shape = {}) {
//        std::vector<int64_t> shape_vec = _batch_shape;
//        shape_vec.insert(shape_vec.end(), _event_shape.begin(), _event_shape.end());
//
//        std::vector<int64_t> extended_shape;
//        extended_shape.insert(extended_shape.end(), sample_shape.begin(), sample_shape.end());
//        extended_shape.insert(extended_shape.end(), shape_vec.begin(), shape_vec.end());
//
//        auto eps = torch::randn(extended_shape, loc_.options());
//        return loc_.expand(shape_vec) + _batch_mv(scale_tril(), eps);
//    }
//
//    torch::Tensor log_prob(const torch::Tensor& value) {
//        auto diff = value - loc_.expand(value.sizes());
//        auto M = _batch_mahalanobis(_unbroadcasted_scale_tril, diff);
//        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
//        return -0.5 * (_event_shape[0] * std::log(2 * M_PI) + M) - half_log_det;
//    }
//
//    torch::Tensor entropy() {
//        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
//        auto H = 0.5 * _event_shape[0] * (1.0 + std::log(2 * M_PI)) + half_log_det;
//        if (_batch_shape.empty()) {
//            return H;
//        }
//        return H.expand(_batch_shape);
//    }
//
//private:
//    torch::Tensor scale_tril() {
//        std::vector<int64_t> shape_vec = _batch_shape;
//        shape_vec.insert(shape_vec.end(), _event_shape.begin(), _event_shape.end());
//        shape_vec.insert(shape_vec.end(), _event_shape.begin(), _event_shape.end());
//        return _unbroadcasted_scale_tril.expand(shape_vec);
//    }
//};