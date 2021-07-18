//
// Created by 周恒 on 2021/6/1.
//

#ifndef PIMTORCH_ONLINE_NEURAL_NETWORK_H
#define PIMTORCH_ONLINE_NEURAL_NETWORK_H
#pragma once
#include <ctime>
#include <torch/torch.h>
#include "pim_linear.h"
#include "visdom.h"

using namespace torch::nn;
using namespace visdom;
namespace F = torch::nn::functional;

class ONN : public torch::nn::Module {
public:
  ONN(int features_size, int max_num_hidden_layers, int qtd_neuron_per_hidden_layer, int n_classes, int batch_size=1,
      double beta=0.99, double lr=0.01, double s=0.2, double freeze_threshold=0.005, int log_interval=1000,
      bool use_cuda=false, bool use_pim=false, const pim_array_pro_config* pim_cfg=nullptr, Visdom *vis=nullptr,
      Json::Value* root_ptr=nullptr)
      : features_size(features_size), max_num_hidden_layers(max_num_hidden_layers),
        qtd_neuron_per_hidden_layer(qtd_neuron_per_hidden_layer), n_classes(n_classes), batch_size(batch_size),
        freeze_threshold(freeze_threshold), log_interval(log_interval), use_pim(use_pim), vis(vis), root_ptr(root_ptr) {
    if (torch::cuda::is_available() && use_cuda) {
      device = at::kCUDA;
    }
    this->beta = register_parameter("beta", torch::full({}, beta), false).to(device);
    this->n = register_parameter("lr", torch::full({}, lr), false).to(device);
    this->s = register_parameter("s", torch::full({}, s), false).to(device);

    if (!use_pim) {
      hidden_layers->push_back(Linear(features_size, qtd_neuron_per_hidden_layer));
      for (int i = 0; i < max_num_hidden_layers - 1; i++) {
        hidden_layers->push_back(Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer));
      }
      for (int i = 0; i < max_num_hidden_layers; i++) {
        output_layers->push_back(Linear(qtd_neuron_per_hidden_layer, n_classes));
        freeze_steps.push_back(0);
      }
    } else {
      hidden_layers->push_back(PimLinear(features_size, qtd_neuron_per_hidden_layer,
                                         batch_size, PimArrayType::pim_array_pro, pim_cfg, false, device));
      for (int i = 0; i < max_num_hidden_layers - 1; i++) {
        hidden_layers->push_back(PimLinear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer,
                                           batch_size, PimArrayType::pim_array_pro, pim_cfg, false, device));
      }
      for (int i = 0; i < max_num_hidden_layers; i++) {
        output_layers->push_back(PimLinear(qtd_neuron_per_hidden_layer, n_classes,
                                           batch_size, PimArrayType::pim_array_pro, pim_cfg, false, device));
        freeze_steps.push_back(0);
      }
    }

    hidden_layers->to(device);
    output_layers->to(device);

    alpha = register_parameter("alpha", torch::empty({max_num_hidden_layers})
        .fill_(1.0 / (max_num_hidden_layers + 1)), false).to(device);

    criterion->to(device);
    cumulative_error = 0;
    batch_loss = 0.0;
    wins = std::vector<std::string>(3);
    if (vis) {
      wins[0] = vis->line(torch::zeros({1}), torch::full({1}, 0));  // loss line
      wins[1] = vis->line(torch::zeros({1}), torch::full({1}, 0));  // acc line
      wins[2] = vis->line(torch::zeros({1}), torch::full({1}, 0));  // alpha line
    }
    if (root_ptr) {
      loss_ptr = new Json::Value(Json::arrayValue);
      error_ptr = new Json::Value(Json::arrayValue);
      alpha_ptr = new Json::Value(Json::arrayValue);
    }
  }

  void zero_grad() override {
    for (int i = 0; i < max_num_hidden_layers; i++) {
      output_layers[i]->zero_grad();
      hidden_layers[i]->zero_grad();
    }
  }

  template<typename LayerType>
  void update_weight(torch::Tensor& X, torch::Tensor& Y, int batch_idx, bool show_loss) {
    torch::Tensor predictions_per_layer = this->forward(X);
    std::vector<torch::Tensor> losses_per_layer(max_num_hidden_layers);
    std::vector<torch::Tensor> w(max_num_hidden_layers);
    std::vector<torch::Tensor> b(max_num_hidden_layers);

    for (int i = 0; i < max_num_hidden_layers; i++) {
      losses_per_layer[i] = this->criterion->forward(predictions_per_layer[i], Y);
    }

    {
      torch::NoGradGuard no_grad;
      for (int i = 0; i < max_num_hidden_layers; i++) {
        losses_per_layer[i].backward({}, true);
        at::Tensor delta_w = n * alpha[i] * output_layers[i]->as<LayerType>()->weight.grad().data();
        at::Tensor delta_b = n * alpha[i] * output_layers[i]->as<LayerType>()->bias.grad().data();
        output_layers[i]->as<LayerType>()->weight.data() -= delta_w;
        output_layers[i]->as<LayerType>()->bias.data() -= delta_b;

        for (int j = 0; j < i + 1; j++) {
          if (!w[j].defined()) {
            w[j] = alpha[i] * hidden_layers[j]->as<LayerType>()->weight.grad().data();
            b[j] = alpha[i] * hidden_layers[j]->as<LayerType>()->bias.grad().data();
          } else {
            w[j] += alpha[i] * hidden_layers[j]->as<LayerType>()->weight.grad().data();
            b[j] += alpha[i] * hidden_layers[j]->as<LayerType>()->bias.grad().data();
          }
        }
        this->zero_grad();
      }

      for (int i = 0; i < max_num_hidden_layers; i++) {
        if (alpha[i].item<double>() >= freeze_threshold || batch_idx % 2 == 0) {
          at::Tensor delta_w = n * w[i];
          at::Tensor delta_b = n * b[i];
          hidden_layers[i]->as<LayerType>()->weight.data() -= delta_w;
          hidden_layers[i]->as<LayerType>()->bias.data() -= delta_b;
        } else {
          freeze_steps[i] += 1;
        }
      }

      for (int i = 0; i < max_num_hidden_layers; i++) {
        alpha[i] *= torch::pow(this->beta, losses_per_layer[i]);
        alpha[i] = torch::max(alpha[i], this->s / max_num_hidden_layers);
      }
    }

    at::Tensor z_t = torch::sum(alpha);
    alpha /= z_t;

    at::Tensor real_output = torch::sum(torch::mul(alpha.view({max_num_hidden_layers, 1}).repeat({1, batch_size})
        .view({max_num_hidden_layers, batch_size, 1}), predictions_per_layer), 0);
    cumulative_error += torch::argmax(real_output, 1).ne(Y).sum().item<double>();

    if (show_loss && batch_idx % log_interval == 0) {
      at::Tensor loss = criterion(real_output, Y);
      batch_loss += loss.template item<double>();
      std::cout << "\nTraining loss: " << batch_loss << ", "
                << "Cumulative error: " << cumulative_error / (batch_idx + 1) * batch_size << std::endl;
      if (vis) {
        vis->line(torch::full({1}, batch_loss), torch::full({1}, batch_idx),
                             wins[0], {}, UpdateMethod::Append);
        vis->line(torch::full({1}, cumulative_error / (batch_idx + 1) * batch_size),
                  torch::full({1}, batch_idx),
                  wins[1], {}, UpdateMethod::Append);
        for (int i = 0; i < max_num_hidden_layers; i++) {
          vis->line(torch::full({1}, alpha[i].item()), torch::full({1}, batch_idx), wins[2],
                    "training_onn","layer" + to_string(i),{}, UpdateMethod::Append);
        }
      }
      if (root_ptr) {
        Json::Value point;
        Json::Value alpha_vec({1, 2, 3});
        point["x"] = batch_idx;
        point["y"] = batch_loss;
        (*loss_ptr).append(point);

        point["y"] = cumulative_error / (batch_idx + 1) * batch_size;
        (*error_ptr).append(point);

        for (int i = 0; i < max_num_hidden_layers; i++) {
          alpha_vec.append(alpha[i].template item<double>());
        }
        point["y"] = alpha_vec;
        alpha_ptr->append(point);
      }
      batch_loss = 0.0;
    }
  }

  torch::Tensor forward(torch::Tensor& X) {
    std::vector<torch::Tensor> hidden_connections, output_class;
    if (!use_pim) {
      torch::Tensor x = F::relu(hidden_layers[0]->as<Linear>()->forward(X));
      hidden_connections.push_back(x);
      for (int i = 1; i < max_num_hidden_layers; i++) {
        hidden_connections.push_back(F::relu(hidden_layers[i]->as<Linear>()->forward(hidden_connections[i-1])));
      }
      for (int i = 0; i < max_num_hidden_layers; i++) {
        output_class.push_back(output_layers[i]->as<Linear>()->forward(hidden_connections[i]));
      }
    } else {
      torch::Tensor x = F::relu(hidden_layers[0]->as<PimLinear>()->forward(X));
      hidden_connections.push_back(x);
      for (int i = 1; i < max_num_hidden_layers; i++) {
        hidden_connections.push_back(F::relu(hidden_layers[i]->as<PimLinear>()->forward(hidden_connections[i-1])));
      }
      for (int i = 0; i < max_num_hidden_layers; i++) {
        output_class.push_back(output_layers[i]->as<PimLinear>()->forward(hidden_connections[i]));
      }
    }
    torch::Tensor pred_per_layer = torch::stack(output_class);
    return pred_per_layer;
  }

  void partial_fit(torch::Tensor& X, torch::Tensor& Y, int batch_idx, bool show_loss) {
    TORCH_CHECK(X.sizes().size() == 2, "Wrong dimension for this X data. It should have only two dimensions.");
    TORCH_CHECK(Y.sizes().size() == 1, "Wrong dimension for this Y data. It should have only one dimensions.");
    if (!use_pim) {
      update_weight<Linear>(X, Y, batch_idx, show_loss);
    } else {
      update_weight<PimLinear>(X, Y, batch_idx, show_loss);
    }
  }

  torch::Tensor predict(torch::Tensor& X) {
    TORCH_CHECK(X.sizes().size() == 2, "Wrong dimension for this X data. It should have only two dimensions.");
    return torch::argmax(torch::sum(torch::mul(alpha.view({max_num_hidden_layers, 1})
              .repeat({1, X.size(0)}).view({max_num_hidden_layers, X.size(0), 1}),
                                               forward(X)), 0), 1);
  }

  void save_params(const std::string path) {

  }

  void load_params(const std::string path) {

  }

  ~ONN() {
    if (root_ptr) {
      (*root_ptr)["loss"] = *loss_ptr;
      (*root_ptr)["error"] = *error_ptr;
      (*root_ptr)["alpha"] = *alpha_ptr;
    }
    delete loss_ptr;
    delete error_ptr;
    delete alpha_ptr;
  }

public:
  int features_size;
  int max_num_hidden_layers;
  int qtd_neuron_per_hidden_layer;
  int n_classes;
  int batch_size;
  torch::Tensor alpha;
  at::Tensor beta;
  at::Tensor n;
  at::Tensor s;
  torch::nn::ModuleList hidden_layers;
  torch::nn::ModuleList output_layers;
  torch::nn::CrossEntropyLoss criterion;

  torch::Device device = at::kCPU;
  double freeze_threshold;
  std::vector<int> freeze_steps;
  double batch_loss;
  double cumulative_error;
  int log_interval;
  bool use_pim;
  Visdom *vis;
  Json::Value* root_ptr;
  Json::Value* loss_ptr;
  Json::Value* error_ptr;
  Json::Value* alpha_ptr;
  std::vector<std::string> wins;
};


#endif //PIMTORCH_ONLINE_NEURAL_NETWORK_H
