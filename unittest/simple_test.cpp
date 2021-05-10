#include <torch/torch.h>
#include <iostream>

using namespace torch;

using std::cout;
using std::endl;
using torch::indexing::Slice;

int main()
{
    std::vector<double> mat = std::vector<double>({0, 2, 3, 1, 8, 4, 5, 7, 9});
    Tensor target = torch::from_blob(mat.data(), {3, 3}, torch::kDouble);
    auto part = target.index({1, Slice(0, 0)});
    cout << target << endl << part << endl;
//    double minconduct = 1;
//    double delta = 2;
//    std::vector<double> vec = std::vector<double>({-1, 2, 3, -3, 0, 0, 1, -2});
//    std::vector<double> mat = std::vector<double>({0, 2, 3, 1});
//    Tensor data = torch::from_blob(vec.data(), {2, 2, 2}, torch::kDouble);
//    Tensor target = torch::from_blob(mat.data(), {2, 2}, torch::kDouble);
//    cout << data << endl <<target << endl;
//    auto data2 = data.pow(2);
//    auto realG = target.mul(delta).add(minconduct);
//    auto ans = data2.matmul(realG);
//
//    cout << data2 << endl << realG << ans << endl;
//
//    auto result = ans.sum();
//    cout << result << endl;
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
