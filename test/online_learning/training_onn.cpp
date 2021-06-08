#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include "visdom.h"
#include "online_neural_network.h"
#include "progressbar.h"
#include "yaml-cpp/yaml.h"


using namespace std::chrono;
using namespace torch::nn;

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


template <typename DataLoader>
void train(
    size_t epoch,
    ONN& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;

  progressbar *bar = progressbar_new("training onn",dataset_size);
  for (auto& batch : data_loader) {
    auto data = batch.data.view({batch.data.size(0), -1}).to(device), targets = batch.target.to(device);
    model.partial_fit(data, targets, batch_idx, true);
    batch_idx++;
    progressbar_inc(bar);
  }
  progressbar_finish(bar);
}

template <typename DataLoader>
void test(
    ONN& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  int32_t correct = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.view({batch.data.size(0), -1}).to(device), targets = batch.target.to(device);
    auto predictions = model.predict(data);
    correct += predictions.eq(targets).sum().template item<int64_t>();
  }
  double accuracy = 100. * correct / dataset_size;
  std::printf("\nTest set: Accuracy: %.3f\n", accuracy);
}


int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: training_onn <yaml config path>\n";
    return -1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  torch::manual_seed(1);
  torch::DeviceType device_type;
  if (torch::cuda::is_available() && config["use_cuda"].as<bool>()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  visdom::Visdom vis(visdom::ConnectionParams("localhost", 8097), "training_onn");

  ONN model(config["feature_size"].as<int>(),
            config["hidden_layers"].as<int>(),
            config["hidden_width"].as<int>(),
            config["classes"].as<int>(),
            config["batch_size"].as<int>(),
            config["beta"].as<double>(),
            config["lr"].as<double>(),
            config["s"].as<double>(),
            config["freeze_threshold"].as<double>(),
            config["log_interval"].as<int>(),
            config["use_cuda"].as<bool>(),
            config["use_pim"].as<bool>(), &vis);
  model.to(device);

  auto start = high_resolution_clock::now();

  auto train_dataset = torch::data::datasets::MNIST(config["data"].as<std::string>())
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
      config["data"].as<std::string>(), torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), 1000);

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, train_dataset_size);
  }
  test(model, device, *test_loader, test_dataset_size);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
}