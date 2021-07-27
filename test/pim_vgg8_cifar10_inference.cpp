//
// Created by 15566 on 2021/7/27.
//

#include "pim_utils.h"
#include <iostream>
#include <string>
#include <ctime>
#include "pim_saver_and_loader.h"
#include "pim_conv.h"
#include "pim_linear.h"

<<<<<<< HEAD
auto runDev = torch::Device(torch::kCUDA, 2);
//auto runDev = torch::Device(torch::kCPU);
int kTestBatchSize = 64;
=======
//auto runDev = torch::Device(torch::kCUDA, 2);
auto runDev = torch::Device(torch::kCPU);
int kBatchSize = 64;
>>>>>>> 07246b11d834270d4143a218cc624de30e62ddf5
//int kTrainBatchSize = 16;
//int kNumberOfEpochs = 10;
std::string out_string = "pim_vgg8_cifar10_out";
auto pim_type = PimArrayType::pim_array_pro;
//auto pim_type = PimArrayType::simple_logic_array;
auto fast_mode = false;
std::string weight_path = "../log/original_vgg8_cifar10.weight";
//todo::need to modify
struct VGG8_Net: torch::nn::Module
{
    VGG8_Net(): conv(5, nullptr), fc1(nullptr), fc2(nullptr), fc3(nullptr)
    {
        //conv[0] = register_module("conv0", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
        conv[0] = register_module("conv0", PimConv2d(ExpandingArray<4>({kBatchSize, 3, 32, 32}), pim_type, Conv2dOptions(3, 64, 3).padding(1), fast_mode, runDev));

        //conv[1] = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        conv[1] = register_module("conv1", PimConv2d(ExpandingArray<4>({kBatchSize, 64, 16, 16}), pim_type, Conv2dOptions(64, 128, 3).padding(1), fast_mode, runDev));

        //conv[2] = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
        conv[2] = register_module("conv2", PimConv2d(ExpandingArray<4>({kBatchSize, 128, 8, 8}), pim_type, Conv2dOptions(128, 256, 3).padding(1), fast_mode, runDev));

        //conv[3] = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
        conv[3] = register_module("conv3", PimConv2d(ExpandingArray<4>({kBatchSize, 256, 4, 4}), pim_type, Conv2dOptions(256, 512, 3).padding(1), fast_mode, runDev));

        //conv[4] = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
        conv[4] = register_module("conv4", PimConv2d(ExpandingArray<4>({kBatchSize, 512, 2, 2}), pim_type, Conv2dOptions(512, 512, 3).padding(1), fast_mode, runDev));


        //fc1 = register_module("fc1", torch::nn::Linear(512, 512));
        fc1 = register_module("fc1", PimLinear(512, 512, kBatchSize, pim_type, fast_mode, runDev));

        //fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc2 = register_module("fc2", PimLinear(512, 512, kBatchSize, pim_type, fast_mode, runDev));

        //fc3 = register_module("fc3", torch::nn::Linear(512, 10));
        fc3 = register_module("fc3", PimLinear(512, 10, kBatchSize, pim_type, fast_mode, runDev));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        using torch::relu;
        namespace F = torch::nn::functional;
        x = F::max_pool2d( (relu(conv[0](x))), F::MaxPool2dFuncOptions(2).stride(2));

        x = F::max_pool2d( (relu(conv[1](x))), F::MaxPool2dFuncOptions(2).stride(2));

        x = F::max_pool2d( (relu(conv[2](x))), F::MaxPool2dFuncOptions(2).stride(2));

        x = F::max_pool2d( (relu(conv[3](x))), F::MaxPool2dFuncOptions(2).stride(2));

        x = F::max_pool2d( (relu(conv[4](x))), F::MaxPool2dFuncOptions(2).stride(2));

        // x = F::max_pool2d( relu(conv[6](x)), F::MaxPool2dFuncOptions(2).stride(2) );
        x = x.view({x.size(0), -1});
        // x = torch::dropout(x, /*p=*/0.6, /*training=*/is_training());
        x = torch::relu(fc1(x));
        //x = torch::dropout(x, /*p=*/0.6, /*training=*/is_training());
        x = torch::relu(fc2(x));
        //x = torch::dropout(x, /*p=*/0.6, /*training=*/is_training());
        x = fc3(x);
        x = torch::log_softmax(x, 1);
        return x;
    }

    // Use one of many "standard library" modules.
    //std::vector<torch::nn::Conv2d> conv;
    std::vector<PimConv2d> conv;
    PimLinear fc1, fc2, fc3;
    //torch::nn::Linear fc1, fc2, fc3;
};


time_t start;
class cifar10Dataset: public torch::data::Dataset<cifar10Dataset>
{
private:
    std::vector<torch::Tensor> images, labels;
public:
    cifar10Dataset(const std::string &path)
    {
        add(path);
    }

    void add(const std::string &path)
    {
        FILE *f = fopen(path.c_str(), "rb");
        if (f==nullptr)
        {
            std::cout << "open fail, path = " << path << std::endl;
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
                std::cout << "read fail, path = " << path << std::endl;
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

template<typename DataLoader>
void mytest(std::shared_ptr<VGG8_Net> &net,
            DataLoader &data_loader,
            torch::Device device,
            size_t data_size
)
{
    int correct = 0;
    //change to inference
    net->eval();
    int i = 0;
    for (auto &batch : data_loader)
    {
        torch::Tensor prediction = net->forward(batch.data.to(device));
        auto out = prediction.argmax(1);
        correct += out.eq(batch.target.to(device).view({-1})).sum().template item<int64_t>();
        std::cout << "batch :" << i << std::endl;
        i++;
    }
    std::cout << "Test datasize= " << data_size << ",  Accuracy: " << 1.0 * correct / data_size << std::endl;
}

//template<typename DataLoader>
//void mytrain(std::shared_ptr<VGG8_Net> &net,
//             DataLoader &data_loader,
//             torch::Device device,
//             size_t data_size,
//             size_t batch_size,
//             torch::optim::Optimizer& optimizer,
//             int epoch//,
//        //bool going_on
//)
//{
//    size_t batch_index = 0;
//    int correct = 0;
//    int ssize = 0;
//    // Iterate the data loader to yield batches from the dataset.
//    for (auto &batch : data_loader)
//    {
//        // Reset gradients.
//        optimizer.zero_grad();
//        // Execute the model on the input data.
//        torch::Tensor prediction = net->forward(batch.data.to(device));
//
//        auto out = prediction.argmax(1);
//        correct += out.eq(batch.target.to(device).view({-1})).sum().template item<int64_t>();
//
//        ssize += batch_size;
//        torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device).to(torch::kLong).view({-1}));
//        // Compute gradients of the loss w.r.t. the parameters of our model.
//        loss.backward();
//        // Update the parameters based on the calculated gradients.
//        optimizer.step();
//        // Output the loss and checkpoint every 100 batches.
//        if (++batch_index % 2 == 0)
//        {
//            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
//                      << " | Loss: " << loss.template item<float>()
//                      << " | correcct = " << correct << " size = " << ssize << " accuracy = " << 1.0*correct/ssize << std::endl;
//            // Serialize your model periodically as a checkpoint.
//            correct = 0, ssize = 0;
//        }
//        if (batch_index%500==0)
//        {
//            // if (!going_on)
//            //     torch::save(net, "net.pt");
//            // else
//            //     torch::save(net, "net_go.pt");
//            time_t now = time(0);
//            std::cout << "Already cost " << difftime(now, start) << " seconds" << std::endl;
//        }
//    }
//}

int main(int argc, char *argv[])
{
    auto net = std::make_shared<VGG8_Net>();
    std::string tr_data_path = "../data/cifar-10-batches-bin/";

//    cifar10Dataset train_data(tr_data_path+"data_batch_1.bin");
//    train_data.add(tr_data_path+"data_batch_2.bin");
//    train_data.add(tr_data_path+"data_batch_3.bin");
//    train_data.add(tr_data_path+"data_batch_4.bin");
//    train_data.add(tr_data_path+"data_batch_5.bin");
//
//    std::cout << "train data read end" << std::endl;


    std::string test_data_path = "../data/cifar-10-batches-bin/";
    cifar10Dataset test_data(test_data_path+"test_batch.bin");
    std::cout << "test data read end" << std::endl;

    if (torch::cuda::is_available() && runDev.is_cuda())
    {
        std::cout << "gpu enabled" << std::endl;
    }
    else
    {
        std::cout << "on cpu" << std::endl;
        runDev = torch::Device(torch::kCPU);
    }

    //int batch_size = 64;

    //auto tr_data_loader = torch::data::make_data_loader(train_data.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})).map(torch::data::transforms::Stack<>()), batch_size);
    auto te_data_loader = torch::data::make_data_loader(test_data.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})).map(torch::data::transforms::Stack<>()), kBatchSize);

    //torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    net->to(runDev);

    PIM::pim_loader(*net, weight_path);
//    bool going_on = false;
//    if (argc>1 && std::string(argv[1])=="GO_ON")
//    {
//        going_on = true;
//        torch::load(net, "net.pt");
//    }
    start = time(0);

    mytest(net, *te_data_loader, runDev, test_data.size().value());
//    for (int epoch=1; epoch<=50; ++epoch)
//    {
//        //mytrain(net, *tr_data_loader, runDev, train_data.size().value(), batch_size, optimizer, epoch/*, going_on*/);
//        mytest(net, *te_data_loader, runDev, test_data.size().value());
//    }

    time_t now = time(0);
    std::cout << "test finish! Cost " << difftime(now, start) << " seconds" << std::endl;

//    PIM::pim_saver(*net, weight_path);

//    if (!going_on)
//        torch::save(net, "net.pt");
//    else
//        torch::save(net, "net_go.pt");
    return 0;
}
