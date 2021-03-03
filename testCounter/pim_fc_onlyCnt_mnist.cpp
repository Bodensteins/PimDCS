#include <fstream>
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include "../pim_linear.h"
#include "omp.h"

using namespace std::chrono;
using namespace PIM;

// Where to find the MNIST dataset.
const char* kDataRoot = "../data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;
auto runDev = torch::kCPU;
// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
//    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
//    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
//    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    fc1 = register_module("fc1", PimLinear(784, 64, kTrainBatchSize, PimArrayType::only_counters_pim_array, runDev));
    fc2 = register_module("fc2", PimLinear(64, 10, kTrainBatchSize, PimArrayType::only_counters_pim_array, runDev));
    //fc1 = register_module("fc1", PimLinear(784, 64, kTrainBatchSize, PimArrayType::pim_array, runDev));
    //fc2 = register_module("fc2", PimLinear(64, 10, kTrainBatchSize, PimArrayType::pim_array, runDev));
// fc3 = register_module("fc3", PimLinear(32, 10, kTrainBatchSize, PimArrayType::wb_logic_array, runDev));
//    fc1 = register_module("fc1", PimLinear(kTrainBatchSize, PimArrayType::simple_logic_array,
//        LinearOptions(784, 64).bias(false)));
//    fc2 = register_module("fc2", PimLinear(kTrainBatchSize, PimArrayType::simple_logic_array,
//        LinearOptions(64, 32).bias(false)));
//    fc3 = register_module("fc3", PimLinear(kTrainBatchSize, PimArrayType::simple_logic_array,
//        LinearOptions(32, 10).bias(false)));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    // x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc2->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
//  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  PimLinear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  // void pretty_print(std::ostream &stream) const override {
  //   stream << "layer 1" << std::endl;
  //   fc1->pretty_print(stream);
  //   stream << "layer 2" << std::endl;
  //   fc2->pretty_print(stream);
  //   // stream << "layer 3" << std::endl;
  //   // fc3->pretty_print(stream);
  // }
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    std::cout << batch_idx << std::endl;
    auto data = batch.data.to(device, torch::kFloat64), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    pimArrayExampleCounters::phyArrMan.schedule(8, 8*64*64, 16);    
    loss.backward();
    optimizer.step();
    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "Train Epoch: %ld [%5ld/%5ld] Loss: %.4f\n",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
      std::fflush(stdout);
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device, torch::kFloat64), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
        output,
        targets,
        /*weight=*/{},
        torch::Reduction::Sum)
        .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

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

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
      kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  double lr = 0.01;
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(lr).momentum(0.5));

  // torch::optim::Adam optimizer( model.parameters(), torch::optim::AdamOptions(lr));
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
  std::ofstream os("out");
    os << model << std::endl;
  return 0;
}
