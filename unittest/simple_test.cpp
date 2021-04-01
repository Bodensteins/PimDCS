#include <torch/torch.h>
#include <iostream>

using namespace torch;

using std::cout;
using std::endl;
using torch::indexing::Slice;

int main()
{
    torch::Tensor a = torch::tensor({{1.0, 2.0}, {3., 4.}}, torch::requires_grad());
    torch::Tensor b = torch::tensor({{1.0, 2.0}});
    
    cout << a-b << endl;
    auto c = a.detach();

    cout << c.data_ptr() << endl;
    cout << a.data_ptr() << endl;
    cout << c.requires_grad() << endl;
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
