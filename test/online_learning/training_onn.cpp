#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <json/json.h>
#include "visdom.h"
#include "online_neural_network.h"
#include "progressbar.h"
#include "yaml-cpp/yaml.h"


using namespace std::chrono;
using namespace torch::nn;

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The number of epochs to x.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

class NumpyDataset : public torch::data::Dataset<NumpyDataset> {
public:
  NumpyDataset(const std::string &x_path, const std::string &y_path, int64_t features_size, int64_t n_classes) {
    unsigned long long x_size, y_size;
    double elem_x;
    unsigned short elem_y;
    std::ifstream x_ifs(x_path, std::ios::in | std::ios::binary);
    TORCH_CHECK(x_ifs.peek() != EOF, "Train data file is empty!");
    x_ifs.read(reinterpret_cast<char *>(&x_size), sizeof(unsigned long long));
    data_x = std::vector<double>(x_size);
    for (unsigned long long i = 0; i < x_size; i++) {
      x_ifs.read(reinterpret_cast<char *>(&elem_x), sizeof(double));
      data_x[i] = elem_x;
    }
    x_ifs.close();

    std::ifstream y_ifs(y_path, std::ios::in | std::ios::binary);
    TORCH_CHECK(y_ifs.peek() != EOF, "Target data file is empty!");
    y_ifs.read(reinterpret_cast<char *>(&y_size), sizeof(unsigned long long));
    data_y = std::vector<unsigned short >(y_size);
    for (unsigned long long i = 0; i < y_size; i++) {
      y_ifs.read(reinterpret_cast<char *>(&elem_y), sizeof(unsigned short));
      data_y[i] = elem_y;
    }
    y_ifs.close();
    TORCH_CHECK((x_size / features_size) == (y_size / n_classes), "Element size of train data doesn't equal to target data.")
    dataset_size = (int64_t)(x_size / features_size);
    x = torch::from_blob(
        data_x.data(), {(int64_t)x_size}, torch::kDouble).reshape({features_size, (int64_t)(x_size / features_size)}).t();
    y = torch::from_blob(
        data_y.data(), {(int64_t)y_size}, torch::kUInt8).reshape({n_classes, (int64_t)(y_size / n_classes)}).t();
  }

  //override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override
  {
    return {x[index], y[index].argmax()};
  };

  //return the length of the data
  torch::optional<size_t> size() const override
  {
    return dataset_size;
  };
private:
  int64_t dataset_size;
  std::vector<double> data_x;
  std::vector<unsigned short > data_y;
  torch::Tensor x, y;
};


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
//    auto data = batch.data.to(device), targets = batch.target.to(device);
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
//    auto data = batch.data.to(device), targets = batch.target.to(device);
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
  const pim_array_pro_config pim_cfg(config["pim_array_config"].as<std::string>());

  torch::manual_seed(1);
  torch::Device device(torch::kCPU, 0);
  if (torch::cuda::is_available() && config["use_cuda"].as<bool>()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  visdom::Visdom vis(visdom::ConnectionParams("localhost", 8097), "training_onn");
  std::vector<std::string> args_name({
     "dataset_name",
     "j",
     "batch_size",
     "lr",
     "beta",
     "s",
     "hidden_layers",
     "hidden_width",
     "freeze_threshold",
     "feature_size",
     "classes",
  });

  std::stringstream out_str;

  Json::Value root = Json::Value();
  Json::Value arguments = Json::Value();
  Json::Value freeze_res = Json::Value();
  std::ofstream json_out;
  std::unique_ptr<Json::StreamWriter> writer(Json::StreamWriterBuilder().newStreamWriter());
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  out_str << config["json_output_dir"].as<std::string>()
          << "/" << config["dataset"].as<std::string>() << std::put_time(&tm, "_%Y-%m-%d_%H-%M-%S.json");
  json_out.open(out_str.str(), std::ios::out | std::ios::trunc);

  std::string text_win = vis.text("Training arguments:\n");
  for (auto it = args_name.begin(); it != args_name.end(); it++) {
    out_str << *it << ": " << config[*it].as<std::string>() << ", ";
    vis.text(out_str.str(), text_win, "training_onn", {});
    out_str.clear();
    arguments[*it] = config[*it].as<std::string>();
  }
  root["arguments"] = arguments;

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
            config["use_pim"].as<bool>(), &pim_cfg, &vis);
  model.to(device);

  auto start = high_resolution_clock::now();

  // MNIST
  auto train_dataset = torch::data::datasets::MNIST(config["data"].as<std::string>())
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  auto test_dataset = torch::data::datasets::MNIST(
      config["data"].as<std::string>(), torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());

  // numpy
//  auto train_dataset = NumpyDataset(
//      config["data_x"].as<std::string>(),
//      config["data_y"].as<std::string>(),
//      config["feature_size"].as<int>(),
//      config["classes"].as<int>()).map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value();

  auto train_loader = torch::data::make_data_loader(std::move(train_dataset),
                                                    config["batch_size"].as<int>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1000);

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, train_dataset_size);
  }
  test(model, device, *test_loader, test_dataset_size);

  for (int i = 0; i < model.max_num_hidden_layers; i++) {
    out_str << "freeze steps of layer " << i << ": " << model.freeze_steps[i]
            << "(" << (double)model.freeze_steps[i] / train_dataset_size << "), ";
    vis.text(out_str.str(), text_win, "training_onn", {});
    out_str.clear();
    freeze_res.append(model.freeze_steps[i]);
  }
  root["freeze_steps"] = freeze_res;
  root["total_steps"] = (int)train_dataset_size;

  out_str << "total steps: " << train_dataset_size << "\n";
  vis.text(out_str.str(), text_win, "training_onn", {});
  out_str.clear();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time: " << duration.count() / 1000. << " seconds" << std::endl;

  vis.save({"onn_training"});

  writer->write(root, &json_out);
  json_out.close();
}