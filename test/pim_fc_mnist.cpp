#include <fstream>
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include "pim_linear.h"
#include "omp.h"
#include "pim_saver_and_loader.h"

using namespace std::chrono;
using namespace PIM;

// Where to find the MNIST dataset.
const char* kDataRoot = "../data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 100;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 50;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

auto runDev = torch::Device(torch::kCUDA);
//auto runDev = torch::Device(torch::kCPU);
auto fastmode = false;
// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", PimLinear(784, 64, kTrainBatchSize, PimArrayType::pim_array_pro, fastmode, TensorOptions(torch::kF32).device(runDev)));
    fc2 = register_module("fc2", PimLinear(64, 10, kTrainBatchSize, PimArrayType::pim_array_pro, fastmode, TensorOptions(torch::kF32).device(runDev)));
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

bool skip = false;

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
    //std::cout << batch_idx << std::endl;
    auto data = batch.data.to(device, torch::kFloat32), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    // sync weights
    model.apply([](nn::Module& module) {
      PIM::PimLinearImpl* module_ptr = dynamic_cast<PIM::PimLinearImpl*>(&module);
      if (module_ptr) {
        module_ptr->sync_weight();
      }
    });
    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "Train Epoch: %d [%5ld/%5ld] Loss: %.4f\n",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
      std::fflush(stdout);
    }
  }
}

double accArr[55];

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
  static int inTestCnt = 0;

  inTestCnt++;

  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device, torch::kFloat32), targets = batch.target.to(device);
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
  accArr[inTestCnt-1] = 1.0*correct/dataset_size;

  if (inTestCnt>5)
  {
	  bool decrease = true;
	  for (int i=inTestCnt-5; i<inTestCnt; ++i)
	  {
		  if (accArr[i]>accArr[inTestCnt-6])
		{
			decrease = false;
			break;
		}
	  }
	  if (decrease)
	  	skip = true;
  }
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main(int argc, char *argv[]) -> int {
  //torch::manual_seed(1);
  
  if (torch::cuda::is_available() && runDev.is_cuda())
  {
      std::cout << "gpu enabled" << std::endl;
  } 
  else
  {
      std::cout << "on cpu" << std::endl;
      runDev = torch::Device(torch::kCPU);
  }   

  Net model;

  model.to(runDev, torch::kFloat32);

  pimArrayPro::phyArrManPro.printArea(std::cout);
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
  if (argc>1 && std::string(argv[1])=="GO_ON")
  {
	  PIM::pim_loader(model, "net_pim_logicArray_fc_mnist.pt");
      test(model, runDev, *test_loader, test_dataset_size);
  }
  else  
  {
  	for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
	    if (skip) 
  		{
	  		std::cout << "due to accuracy is always decreasing for 5 epochs, training is skip to the end." << std::endl;
	  		return;
  		}
    train(epoch, model, runDev, *train_loader, optimizer, train_dataset_size);
    test(model, runDev, *test_loader, test_dataset_size);
  }
 // PIM::pim_saver(model, "net_pim_logicArray_fc_mnist.pt");
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
  pimArrayPro::phyArrManPro.print_latency(std::cout);
  pimArrayPro::phyArrManPro.print_op(std::cout);
  pimArrayPro::phyArrManPro.print_energy(std::cout);
  pimArrayPro::phyArrManPro.print_power_efficiency(std::cout);
  pimArrayPro::phyArrManPro.print_compute_all_energy(std::cout);
  //std::ofstream of("pim_fc_phyArray_weight.out");
  //pimArrayPro::phyArrManPro.printAllInfo(of);
  //of.close();
  return 0;
}
