import json

# 讀取 graph_c.json
with open('./build/graph_c.json', 'r') as f:
    graph = json.load(f)

# 從 shape 列表中找到輸入和輸出的維度
shapes = graph['attrs']['shape'][1]  # [1] 是因為第一個元素是 "list_shape"

# 輸入維度是第一個 shape
input_shape = shapes[0]
print("Input shape:", input_shape)  # 應該是 [1, 3, 224, 224]

# 輸出維度是最後一個 shape
output_shape = shapes[-1]
print("Output shape:", output_shape)  # 應該是 [1, 1280, 1, 1] 或類似的維度

# 如果你想看所有的維度變化
# print("\nAll shapes in order:")
# for i, shape in enumerate(shapes):
#     print(f"Layer {i}: {shape}")