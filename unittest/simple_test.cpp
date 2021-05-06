#include <torch/torch.h>
#include <iostream>

using namespace torch;

using std::cout;
using std::endl;
using torch::indexing::Slice;

int main()
{
    double minconduct = 0.5;
    double delta = 15;
    std::vector<int> vec = std::vector<int>({-1, 2, 3, -3, 0, 0, 1, -2});
    std::vector<int> mat = std::vector<int>({0, 2, 3, 1});
    Tensor data = torch::from_blob(vec.data(), {2, 2, 2}, torch::kInt);
    Tensor target = torch::from_blob(mat.data(), {2, 2}, torch::kInt);
    cout << data << endl <<target << endl;
    auto data2 = data.pow(2);
    auto ans = data2.matmul(target);

    cout << data2 << endl << ans << endl;

    auto result = ans.sum(0);
    cout << result
//    torch::Tensor a = torch::tensor({{1.0, 2.0}, {3., 4.}}, torch::requires_grad());
//    torch::Tensor b = torch::tensor({{1.0, 2.0}});
//
//    cout << a-b << endl;
//    auto c = a.detach();
//
//    cout << c.data_ptr() << endl;
//    cout << a.data_ptr() << endl;
//    cout << c.requires_grad() << endl;
    // Tensor k = torch::tensor({{2.0, 3.0}, {1.4, 0.7}}).to(torch::kI32);
    // //cout << k << endl;
    // Tensor myindex = k > 1;
    // at::Tensor output = torch::empty({k.size(0), 2, k.size(1)}, TensorOptions(k.device()).dtype(torch::kI32));
    // at::Tensor tindex = torch::empty({k.size(0), 2, k.size(1)}, TensorOptions(k.device()).dtype(torch::kBool));
    
    //     for (int i = 0; i < 2; ++i)
    //     {
    //         output.index_put_({Slice(), Slice(i, i + 1)}, (k.__rshift__(i).bitwise_and(1)).view({k.size(0), 1, k.size(1)}) );
    //         tindex.index_put_({Slice(), Slice(i, i + 1)}, myindex.view({k.size(0), 1, k.size(1)}));
    //     }
    
    // cout << output << endl;
    // cout << tindex << endl;
    // output.index_put_({tindex}, output.index({tindex})*-1);
    // cout << output << endl;
    return 0;
}
