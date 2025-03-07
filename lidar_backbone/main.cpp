#include "lidar_backbone.h"

int main() {
    // 创建输入 - 使用正确的空间尺寸
    auto indices = torch::tensor({
        {0, 0, 0},      // batch index
        {720, 720, 720}, // z
        {720, 720, 720}, // y
        {20, 20, 20}     // x
    }, torch::kLong);
    
    auto values = torch::randn({3, 5}); // 5个输入通道
    std::vector<int64_t> spatial_size = {1440, 1440, 41}; // 初始空间尺寸

    // 创建模型
    auto model = LidarBackbone();
    
    // 前向传播
    auto output = model->forward(indices, values, spatial_size);
    
    // 验证输出形状是否符合预期 [1, 256, 180, 180]
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    return 0;
}
