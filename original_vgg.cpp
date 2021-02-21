#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include "vgg.h"

using namespace std::chrono;
using namespace torch::nn;

// Where to find the CIFAR10 dataset.
//std::string kDataRoot = "../data/cifar10-dataset/";
std::string kDataRoot = "../data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 40;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

class CIFAR10Dataset : public torch::data::Dataset<CIFAR10Dataset>
{
private:
  std::vector<torch::Tensor> images, labels;
public:
  CIFAR10Dataset(const std::string &path)
  {
    add(path);
  }

  void add(const std::string &path)
  {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr)
    {
      std::cout << "Cannot open " << path << std::endl;
      return;
    }

    unsigned char buf[4096];
    std::vector<float> data(3072);
    std::cout << "read file: " << path << std::endl;
    std::cout << '[';
    for (int i=0; i<98; ++i)
      std::cout << '-';
    std::cout << ']' << std::endl;

    for (int i=0; i<10000; ++i)
    {
      if (fread(buf, 3073, 1, f)==-1)
      {
        std::cout << "Cannot open" << path << std::endl;
        return;
      }
      if (i%100==0)
      {
        std::cout << '*';
        std::cout << std::flush;
      }
      torch::Tensor tmp = torch::empty({1});
      tmp[0] = (int)buf[0];
      labels.push_back(tmp);

      for (int j=0; j<3072; ++j)
        data[j] = (int)buf[j];
      // if (i<=2)
      // {
      //     for (int j=0; j<10; ++j)
      //         std::cout << data[j] << ' ' << std::endl;
      // }
      auto tharray = torch::zeros({3, 32, 32}, torch::kFloat);
      std::memcpy(tharray.data_ptr(), data.data(), sizeof(float)*3072);
      tharray.div_(255);
      images.push_back(tharray);
    }
    std::cout << std::endl;
    fclose(f);
  }

  //override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override
  {
    return { images[index].clone(), labels[index].clone() };
  };

  //return the length of the data
  torch::optional<size_t> size() const override
  {
    return labels.size();
  };
};


template <typename DataLoader>
void train(
    size_t epoch,
    VGG& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device, torch::kFloat64), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, targets);

    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    VGG& model,
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
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);


  std::vector<std::array<int, 2>> conv_arch_shape = {
      {1, 64},
      {1, 128},
      {2, 256},
//      {2, 512},
//      {2, 512},
  };
  VGG model(conv_arch_shape);
  model.to(device, torch::kFloat64);

  auto start = high_resolution_clock::now();

//  std::cout << "Reading data..." << std::endl;
//  CIFAR10Dataset train_data(kDataRoot + "data_batch_1.bin");
//  train_data.add(kDataRoot + "data_batch_2.bin");
//  train_data.add(kDataRoot + "data_batch_3.bin");
//  train_data.add(kDataRoot + "data_batch_4.bin");
//  train_data.add(kDataRoot + "data_batch_5.bin");
//  CIFAR10Dataset test_data(kDataRoot + "test_batch.bin");
//
//  auto train_dataset = train_data.map(
//          torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
//      .map(torch::data::transforms::Stack<>());
//  const size_t train_dataset_size = train_dataset.size().value();
//  auto train_loader =
//      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//          std::move(train_dataset), kTrainBatchSize);
//
//  auto test_dataset = test_data.map(
//          torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
//      .map(torch::data::transforms::Stack<>());
//  const size_t test_dataset_size = test_dataset.size().value();
//  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

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


  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.3));
//  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
}
