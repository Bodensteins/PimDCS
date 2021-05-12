#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yaml-cpp/yaml.h"
#include "Darknet.h"
#include "pim_conv.h"
#include "pim_linear.h"

using namespace std;
using namespace std::chrono;

void detect(Darknet &net, torch::Device &device, YAML::Node &config, int input_image_size, const std::string &img_path,
            const std::string &output_path) {
  cv::Mat origin_image, resized_image;

  origin_image = cv::imread(img_path);

  cv::cvtColor(origin_image, resized_image, cv::COLOR_BGR2RGB);
  cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

  cv::Mat img_float;
  resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

  auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
  img_tensor = img_tensor.permute({0, 3, 1, 2});

  auto start = std::chrono::high_resolution_clock::now();

  auto output = net.forward(img_tensor);

  // filter result by NMS
  // class_num = 80
  // confidence = 0.6
  auto result = net.write_results(output, 80, 0.6, 0.4);

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(end - start);

  // It should be known that it takes longer time at first time
  std::cout << "inference taken : " << duration.count() << " ms" << endl;

  if (result.dim() == 1) {
    std::cout << "no object found" << endl;
  } else {
    int obj_num = result.size(0);

    std::cout << obj_num << " objects found" << endl;

    float w_scale = float(origin_image.cols) / input_image_size;
    float h_scale = float(origin_image.rows) / input_image_size;

    result.select(1, 1).mul_(w_scale);
    result.select(1, 2).mul_(h_scale);
    result.select(1, 3).mul_(w_scale);
    result.select(1, 4).mul_(h_scale);

    std::cout << result << std::endl;
    auto result_data = result.accessor<double, 2>();

    for (int i = 0; i < result.size(0); i++) {
      cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]),
                    cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
      cv::putText(origin_image,
                  config["categories"][int(result_data[i][7])].as<string>() + ", " + to_string(result_data[i][6]),
                  cv::Point(result_data[i][1], result_data[i][2]-3),
                  cv::FONT_HERSHEY_COMPLEX,
                  0.5,
                  cv::Scalar(0, 0, 255),
                  1, 8, 0);
    }

    cv::imwrite(output_path, origin_image);
  }
}


int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: yolo_v3 <yaml config path>\n";
    return -1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  torch::DeviceType device_type;

  constexpr torch::DeviceType runDev = getRunDev();
  if (torch::cuda::is_available() && runDev == torch::kCUDA) {
    device_type = torch::kCUDA;
  } else {
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // input image size for YOLO v3
  int input_image_size = 416;

  Darknet net(config, device);

  map<string, string> *info = net.get_net_info();

  info->operator[]("height") = std::to_string(input_image_size);

  std::cout << "loading weight ..." << endl;
  net.load_weights();
  std::cout << "weight loaded ..." << endl;

  net.to(torch::kF64);
  net.to(device);

  torch::NoGradGuard no_grad;
  net.eval();

  if (config["single_image"].as<bool>()) {
    detect(net, device, config, input_image_size,
           config["img_path"].as<string>(), "detect_out.jpg");
  } else {
    DIR *root = opendir(config["img_dir"].as<string>().c_str());
    struct dirent *ent;
    if (root != NULL) {
      while ((ent = readdir(root)) != NULL) {
        if ((strcmp(ent->d_name, ".") == 0) ||
            (strcmp(ent->d_name, "..") == 0)) {
          continue;
        }

        std::string img_path = config["img_dir"].as<string>() + "/" + ent->d_name;
        std::string out_path = config["output_dir"].as<string>() + "/" + ent->d_name;
        std::cout << "Detecting: " << img_path << std::endl;
        std::cout << "Output: " << out_path << std::endl;
        detect(net, device, config, input_image_size,
               img_path, out_path);
      }
    } else {
      std::cout << "Error! Cannot open image directory." << std::endl;
    }
  }

  pimArrayPro::phyArrManPro.printArea();
  pimArrayPro::phyArrManPro.print_latency(std::cout);
  std::cout << "Total energy: " << pimArrayPro::phyArrManPro.get_total_energy() << " J" << std::endl;

  return 0;
}
