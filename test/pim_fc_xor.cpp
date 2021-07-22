#include <fstream>
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include "pim_linear.h"
#include "omp.h"

using namespace std::chrono;
using namespace PIM;

// Where to find the MNIST dataset.
//const char* kDataRoot = "../data";

// The batch size for training.
const int64_t kTrainBatchSize = 4;

// The batch size for testing.
const int64_t kTestBatchSize = 4;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 100;
auto runDev = torch::kCPU;


const int trainTimes = 1000;
const int train_dataset_size = 4 * trainTimes;

const int testTimes = 1;
const int test_dataset_size = 4 * testTimes;
// Define a new Module.
struct Net : torch::nn::Module {
  Net()
  {
    //fc = register_module("fc", PimLinear(2, 2, kTrainBatchSize, PimArrayType::pim_array_pro, runDev));
    //fc1 = register_module("fc1", torch::nn::Linear(2, 10));
    //fc2 = register_module("fc2", torch::nn::Linear(10, 2));
    fc1 = register_module("fc1", PimLinear(2, 3, kTrainBatchSize, PimArrayType::pim_array_pro, false, runDev));
    fc2 = register_module("fc2", PimLinear(3, 2, kTrainBatchSize, PimArrayType::pim_array_pro, false, runDev));
    //fc = register_module("fc", torch::nn::Linear(2, 2));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1->forward(x));
    x = torch::log_softmax(fc2->forward(x), /*dim=*/1);
    //x = torch::log_softmax(fc->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  //PimLinear fc{nullptr};
  //torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  PimLinear fc1{nullptr}, fc2{nullptr};
  //torch::nn::Linear fc{nullptr};
};

void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();

  std::vector<double> in = std::vector<double>({0, 0, 0, 1, 1, 0, 1, 1});
  std::vector<long> out = std::vector<long>({0, 1, 1, 0});
  Tensor data = torch::from_blob(in.data(), {4, 2}, torch::kF64);
  Tensor target = torch::from_blob(out.data(), {4}, torch::kLong);
  //cout << target << endl;

  for (int i = 1; i <= trainTimes; ++i) {
    data = data.to(device);
    target = target.to(device);

    optimizer.zero_grad();

    auto output = model.forward(data);
    //cout << output << target << endl;
    auto loss = torch::nll_loss(output, target);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (i % kLogInterval == 0) {
      std::printf(
          "Train Epoch: %d [%5ld/%5ld] Loss: %.4f\n",
          epoch,
          i * kTrainBatchSize,
          dataset_size,
          loss.item<float>());
      std::fflush(stdout);
    }
  }
}

void test(
    Net& model,
    torch::Device device,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;

  std::vector<double> in = std::vector<double>({0, 0, 0, 1, 1, 0, 1, 1});
  std::vector<long> out = std::vector<long>({0, 1, 1, 0});
  Tensor data = torch::from_blob(in.data(), {4, 2}, torch::kF64);
  Tensor target = torch::from_blob(out.data(), {4}, torch::kLong);

  auto output = model.forward(data);
  test_loss += torch::nll_loss(
      output,
      target,
      /*weight=*/{},
      torch::Reduction::Sum)
      .template item<float>();
  auto pred = output.argmax(1);
  //cout << output << target << endl;
  correct += pred.eq(target).sum().item<int64_t>();

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (runDev == torch::kCUDA && torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
    runDev = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device, torch::kFloat64);

  auto start = high_resolution_clock::now();

  double lr = 0.1;
  //torch::optim::SGD optimizer(
  //      model.parameters(), torch::optim::SGDOptions(lr).momentum(0.5));

  torch::optim::Adam optimizer(
      model.parameters(), torch::optim::AdamOptions(lr));


  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    train(epoch, model, device, optimizer, train_dataset_size);
    test(model, device, test_dataset_size);
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
  std::ofstream os("out");
  os << model << std::endl;
  return 0;
}