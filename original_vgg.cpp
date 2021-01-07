#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>

using namespace std::chrono;
using namespace torch::nn;

// Where to find the CIFAR10 dataset.
std::string kDataRoot = "/Users/zhouheng/Downloads/cifar10-dataset/";

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

struct VGG : torch::nn::Module {
  VGG() : conv_list(13, nullptr), max_pooling_list(14, nullptr), bn_list(14, nullptr), dropout_list(12, nullptr){
    std::vector<int> in_channels = {3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512};
    std::vector<int> in_shape = {32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2};
    for (int i = 0; i < 13; i++) {
      conv_list[i] = register_module("Conv2d"+std::to_string(i+1),
          Conv2d(Conv2dOptions(in_channels[i], in_channels[i+1], 3).padding(1)));
    }

    for (int i = 0; i < 14; i++) {
      max_pooling_list[i] = register_module("MaxPool2d"+std::to_string(i+1),
          MaxPool2d(MaxPool2dOptions({2, 2})));
    }

    for (int i = 1; i < 15; i++) {
      bn_list[i-1] = register_module("BatchNorm2d"+std::to_string(i),
          BatchNorm2d(BatchNorm2dOptions(in_channels[i])));
    }

    std::vector<double> p = {0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5};
    for (int i = 0; i < 12; i++) {
      dropout_list.emplace_back();
      dropout_list[i] = register_module("Dropout2d"+std::to_string(i+1),
          Dropout2d(Dropout2dOptions().p(p[i])));
    }

    fc1 = register_module("fc1", Linear(512, 4096));
    fc2 = register_module("fc2", Linear(4096, 4096));
    fc3 = register_module("fc3", Linear(4096, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    // layer 1, 64
    x = torch::relu(conv_list[0]->forward(x));
    x = bn_list[0]->forward(x);
    x = dropout_list[0]->forward(x);

    // layer 2, 64
    x = torch::relu(conv_list[1]->forward(x));
    x = bn_list[1]->forward(x);
    x = dropout_list[1]->forward(x);

    // Maxpool
    x = max_pooling_list[0]->forward(x);

    // layer 3, 128
    x = torch::relu(conv_list[2]->forward(x));
    x = bn_list[2]->forward(x);
    x = dropout_list[2]->forward(x);

    // layer 4, 128
    x = torch::relu(conv_list[3]->forward(x));
    x = bn_list[3]->forward(x);
    x = dropout_list[3]->forward(x);

    // Maxpool
    x = max_pooling_list[1]->forward(x);

    // layer 5, 256
    x = torch::relu(conv_list[4]->forward(x));
    x = bn_list[4]->forward(x);
    x = dropout_list[4]->forward(x);

    // layer 6, 256
    x = torch::relu(conv_list[5]->forward(x));
    x = bn_list[5]->forward(x);
    x = dropout_list[5]->forward(x);

    // layer 7, 256
    x = torch::relu(conv_list[6]->forward(x));
    x = bn_list[6]->forward(x);
    x = dropout_list[6]->forward(x);

    // Maxpool
    x = max_pooling_list[2]->forward(x);

    // layer 8, 512
    x = torch::relu(conv_list[7]->forward(x));
    x = bn_list[7]->forward(x);
    x = dropout_list[7]->forward(x);

    // layer 9, 512
    x = torch::relu(conv_list[8]->forward(x));
    x = bn_list[8]->forward(x);
    x = dropout_list[8]->forward(x);

    // layer 10, 512
    x = torch::relu(conv_list[9]->forward(x));
    x = bn_list[9]->forward(x);
    x = dropout_list[9]->forward(x);

    // Maxpool
    x = max_pooling_list[3]->forward(x);

    // layer 11, 512
    x = torch::relu(conv_list[10]->forward(x));
    x = bn_list[10]->forward(x);
    x = dropout_list[10]->forward(x);

    // layer 12, 512
    x = torch::relu(conv_list[11]->forward(x));
    x = bn_list[11]->forward(x);
    x = dropout_list[11]->forward(x);

    // layer 13, 512
    x = torch::relu(conv_list[12]->forward(x));
    x = bn_list[12]->forward(x);

    // Maxpool
    x = max_pooling_list[4]->forward(x);
    x = dropout_list[12]->forward(x);

    // fc1
    x = x.view({-1, 512});
    x = torch::relu(fc1->forward(x));
    x = dropout_list[13]->forward(x);
    x = torch::relu(fc2->forward(x));
    x = dropout_list[14]->forward(x);
    x = torch::relu(fc3->forward(x));
    return torch::log_softmax(x, 1);
  }

  std::vector<torch::nn::Conv2d> conv_list;
  std::vector<torch::nn::BatchNorm2d> bn_list;
  std::vector<torch::nn::Dropout2d> dropout_list;
  std::vector<torch::nn::MaxPool2d> max_pooling_list;
  torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr;
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
    auto loss = torch::nll_loss(output, targets.to(torch::kLong).view({-1}));
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

  VGG model;
  model.to(device, torch::kFloat64);

  auto start = high_resolution_clock::now();

  std::cout << "Reading data..." << std::endl;
  CIFAR10Dataset train_data(kDataRoot + "data_batch_1.bin");
  train_data.add(kDataRoot + "data_batch_2.bin");
  train_data.add(kDataRoot + "data_batch_3.bin");
  train_data.add(kDataRoot + "data_batch_4.bin");
  train_data.add(kDataRoot + "data_batch_5.bin");
  CIFAR10Dataset test_data(kDataRoot + "test_batch.bin");

  auto train_dataset = train_data.map(
          torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = test_data.map(
          torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.0035).momentum(0.3));
//  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;
}
