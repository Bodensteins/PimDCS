#include "pure_array.h"
#include <iostream>
#include <string>
#include <ctime>

auto runDev = torch::kCPU;

struct VGG8_Net: torch::nn::Module
{
    VGG8_Net(): conv(6, nullptr), fc1(nullptr), fc2(nullptr)
    {
        conv[0] = register_module("conv0", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 128, 3).padding(1)));
        conv[1] = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
        conv[2] = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
        conv[3] = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        conv[4] = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
        conv[5] = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
        // conv[6] = register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).padding(1)));
        fc1 = register_module("fc1", torch::nn::Linear(8192, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        using torch::relu;      
        namespace F = torch::nn::functional;
        x = F::max_pool2d( relu(conv[1](relu(conv[0](x)))), F::MaxPool2dFuncOptions(2).stride(2) ); 

        x = F::max_pool2d( relu(conv[3](relu(conv[2](x)))), F::MaxPool2dFuncOptions(2).stride(2) );  

        x = F::max_pool2d( relu(conv[5](relu(conv[4](x)))), F::MaxPool2dFuncOptions(2).stride(2) ); 

        // x = F::max_pool2d( relu(conv[6](x)), F::MaxPool2dFuncOptions(2).stride(2) );
        x = x.view({x.size(0), -1});
        x = torch::dropout(x, /*p=*/0.6, /*training=*/is_training());
        x = torch::relu(fc1(x));
        x = torch::dropout(x, /*p=*/0.6, /*training=*/is_training());
        x = fc2(x);
        x = torch::log_softmax(x, 1);
        return x;
    }

    // Use one of many "standard library" modules.
    std::vector<torch::nn::Conv2d> conv;
    torch::nn::Linear fc1, fc2;
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

    for (auto &batch : data_loader)
    {
         torch::Tensor prediction = net->forward(batch.data.to(device));
         auto out = prediction.argmax(1);
         correct += out.eq(batch.target.to(device).view({-1})).sum().template item<int64_t>();
        
    }
    std::cout << "Test datasize= " << data_size << ",  Accuracy: " << 1.0 * correct / data_size << std::endl;
}

template<typename DataLoader>
void mytrain(std::shared_ptr<VGG8_Net> &net, 
            DataLoader &data_loader,
            torch::Device device,
            size_t data_size,
            size_t batch_size,
            torch::optim::Optimizer& optimizer,
            int epoch,
            bool going_on
            )
{
    size_t batch_index = 0;
    int correct = 0;
    int ssize = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto &batch : data_loader)
    {
        // Reset gradients.
        optimizer.zero_grad();
        // Execute the model on the input data.
        torch::Tensor prediction = net->forward(batch.data.to(device));

        auto out = prediction.argmax(1);
        correct += out.eq(batch.target.to(device).view({-1})).sum().template item<int64_t>();
    
        ssize += batch_size;
        torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device).to(torch::kLong).view({-1}));
        // Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward();
        // Update the parameters based on the calculated gradients.
        optimizer.step();
        // Output the loss and checkpoint every 100 batches.
        if (++batch_index % 2 == 0)
        {
            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.template item<float>() 
                        << " | correcct = " << correct << " size = " << ssize << " accuracy = " << 1.0*correct/ssize << std::endl;
            // Serialize your model periodically as a checkpoint.
            correct = 0, ssize = 0;
        }
        if (batch_index%500==0)
        {
            // if (!going_on) 
            //     torch::save(net, "net.pt");
            // else
            //     torch::save(net, "net_go.pt"); 
            time_t now = time(0);
            std::cout << "Already cost " << difftime(now, start) << " seconds" << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    auto net = std::make_shared<VGG8_Net>();
    std::string tr_data_path = "/home/bing/HDD/mysoft/test/cifar10/dataset/";
    
    cifar10Dataset train_data(tr_data_path+"data_batch_1.bin");
    train_data.add(tr_data_path+"data_batch_2.bin");
    train_data.add(tr_data_path+"data_batch_3.bin");
    train_data.add(tr_data_path+"data_batch_4.bin");
    train_data.add(tr_data_path+"data_batch_5.bin");
    
    std::cout << "train data read end" << std::endl;


    std::string test_data_path = "/home/bing/HDD/mysoft/test/cifar10/dataset/";
    cifar10Dataset test_data(test_data_path+"test_batch.bin");
    std::cout << "test data read end" << std::endl;

    torch::DeviceType device = at::kCPU;
    if (torch::cuda::is_available() && runDev!=torch::kCPU)
    {
        std::cout << "gpu enabled" << std::endl;
        device = at::kCUDA;
    }

    int batch_size = 64;
    
    auto tr_data_loader = torch::data::make_data_loader(train_data.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})).map(torch::data::transforms::Stack<>()), batch_size);
    auto te_data_loader = torch::data::make_data_loader(test_data.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})).map(torch::data::transforms::Stack<>()), batch_size);

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    net->to(device);
    bool going_on = false;  
    if (argc>1 && std::string(argv[1])=="GO_ON")
    {
        going_on = true;
        torch::load(net, "net.pt");
    }
    start = time(0);
    for (int epoch=1; epoch<=50; ++epoch)
    {
        mytrain(net, *tr_data_loader, device, train_data.size().value(), batch_size, optimizer, epoch, going_on);
        mytest(net, *te_data_loader, device, test_data.size().value());
    }  

    time_t now = time(0);
    std::cout << "train finish! Cost " << difftime(now, start) << " seconds" << std::endl;
    if (!going_on)
        torch::save(net, "net.pt");
    else
        torch::save(net, "net_go.pt");  
    return 0;
}
