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
std::string kDataRoot = "../data/cifar10-dataset/";
//std::string kDataRoot = "../data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 10;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 250;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

auto my_device = torch::Device(torch::kCUDA);

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

float accuracy (const torch::Tensor& y_hat, const torch::Tensor& y) {
  auto compare = (y_hat.argmax(/*dim=*/ 1) == y);
  compare = compare.to(torch::kFloat32);
  return compare.sum().item<float>();
}

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
  float train_l = 0, train_acc_sum = 0;

  for (auto& batch : data_loader) {
//    auto data = batch.data.to(device, torch::kFloat64), targets = batch.target.squeeze(1).to(device, torch::kLong);
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, targets);

//    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    train_l = loss.template item<float>();
    train_acc_sum += accuracy(output, targets);

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\nTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f, train accuracy: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>(),
          train_acc_sum / (batch_idx * batch.data.size(0)) * 100);
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
  int data_counter = 0;
  int max_counter = 5000000;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device, torch::kFloat32), targets = batch.target.squeeze(1).to(device, torch::kLong);
//    auto data = batch.data.to(device, torch::kFloat64), targets = batch.target.to(device);
    auto output = model.forward(data);

    test_loss += torch::nn::functional::cross_entropy(
        output,
        targets,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum))
        .template item<float>();

    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    data_counter += targets.size(0);
    if (data_counter > max_counter) {
      break;
    }
    std::cout << data_counter << "/" << dataset_size << std::endl;
  }

  test_loss /= data_counter;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / data_counter);
}

void add_bias_noise(VGG& model) {
  auto b = model.classifier_->at<torch::nn::LinearImpl>(2).bias;
  auto noise = torch::normal(b.mean().item<double>(), 0.1, b.sizes());
  std::cout << b << std::endl;
  std::cout << noise << std::endl;
}

auto main() -> int {

  if (torch::cuda::is_available() && my_device.is_cuda())
  {
      std::cout << "gpu enabled" << std::endl;
  } 
  else
  {
      std::cout << "on cpu" << std::endl;
      my_device = torch::Device(torch::kCPU);
  }   

  std::vector<std::array<int, 2>> conv_arch_shape = {
      {1, 64},
      {1, 128},
      {2, 256},
      {2, 512},
      {2, 512},
  };

  std::vector<std::vector<ExpandingArray<4>>> in_shapes = {
      {{kTrainBatchSize, 3, 32, 32}},
      {{kTrainBatchSize, 64, 16, 16}},
      {{kTrainBatchSize, 128, 8, 8}, {kTrainBatchSize, 256, 8, 8}},
      {{kTrainBatchSize, 256, 4, 4}, {kTrainBatchSize, 512, 4, 4}},
      {{kTrainBatchSize, 512, 2, 2}, {kTrainBatchSize, 512, 2, 2}},
  };

  VGG model(conv_arch_shape, true, kTrainBatchSize, in_shapes);
  model.load_params("../model_params");
  model.to(my_device, torch::kFloat32);

  // sync weights
  model.apply([](nn::Module& module) {
    if (PIM::PimLinearImpl* module_ptr = dynamic_cast<PIM::PimLinearImpl*>(&module)) {
      module_ptr->sync_weight();
      std::cout << "linear sync" << std::endl;
//      std::cout << module_ptr->wb_t_ptr.ptr.get()->read_mat() << std::endl;
    } else if (PIM::PimConv2dImpl* module_ptr = dynamic_cast<PIM::PimConv2dImpl*>(&module)) {
      module_ptr->sync_weight();
      std::cout << "conv sync " << std::endl;
//      std::cout << module_ptr->wb_t_ptr.ptr.get()->read_mat() << std::endl;
    }
  });

  auto start = high_resolution_clock::now();

  CIFAR10Dataset test_data(kDataRoot + "test_batch.bin");
  auto test_dataset = test_data.map(
          torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(
      std::move(test_dataset),
      torch::data::DataLoaderOptions(kTestBatchSize).workers(8));

  test(model, my_device, *test_loader, test_dataset_size);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
  return 0;
}