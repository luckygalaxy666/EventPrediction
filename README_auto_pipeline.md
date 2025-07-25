# 自动化事件处理流程

这个自动化流程整合了从ElasticSearch获取数据、生成标签、运行预测模型的完整操作。

## 功能概述

1. **自动配置ES查询**: 根据提供的事件名称和关键词更新`getGraphformEs.py`配置
2. **从ES获取数据**: 自动运行ES查询并保存数据到`tsmc_es_data/`目录
3. **生成标签**: 根据关键词两两组合自动生成标签并保存到`tsmc_label.json`
4. **运行预测模型**: 自动调用预测模型并生成结果图表

## 文件结构

```
├── auto_event_pipeline.py      # 主自动化脚本
├── example_usage.py            # 使用示例
├── run_model_for_check.py      # 修改后的模型运行脚本
├── ElasticSearch/
│   └── getGraphformEs.py       # ES数据获取脚本
├── tsmc_es_data/               # 数据存储目录
├── tsmc_label.json             # 标签文件
└── main.py                     # 预测模型主程序
```

## 使用方法

### 基本用法

```bash
python auto_event_pipeline.py --event_name "事件名称" --entities 实体1 实体2 实体3 ...
```

### 完整参数

```bash
python auto_event_pipeline.py \
    --event_name "台积电获得英伟达AI芯片订单" \
    --entities 台积电 英伟达 AI芯片 半导体 美国 魏哲家 \
    --event_time "2024-12-15" \
    --mode check \
    --timespan 31 \
    --label_file tsmc_label.json
```

### 参数说明

- `--event_name`: 事件名称（必需）
- `--entities`: 关键词实体列表（必需，可多个）
- `--event_time`: 事件时间，格式YYYY-MM-DD（可选，默认"Unknown"）
- `--mode`: 运行模式（可选，默认"check"）
  - `check`: 检查模式
  - `fit`: 拟合模式
  - `init`: 初始化模式
- `--timespan`: 时间跨度（可选，默认"31"）
- `--label_file`: 标签文件路径（可选，默认"tsmc_label.json"）

## 使用示例

### 示例1: 台积电相关事件

```bash
python auto_event_pipeline.py \
    --event_name "台积电获得英伟达AI芯片订单" \
    --entities 台积电 英伟达 AI芯片 半导体 美国 魏哲家 \
    --event_time "2024-12-15"
```

### 示例2: 政治事件

```bash
python auto_event_pipeline.py \
    --event_name "赖清德访问美国" \
    --entities 赖清德 美国 台湾 拜登 中国 \
    --event_time "2024-11-20"
```

### 示例3: 经济事件

```bash
python auto_event_pipeline.py \
    --event_name "台积电在德国建厂" \
    --entities 台积电 德国 欧洲 半导体 魏哲家 \
    --event_time "2024-10-25"
```

## 运行示例脚本

```bash
python example_usage.py
```

这将显示所有使用示例，并询问是否要运行第一个示例。

## 输出结果

运行完成后，您将得到：

1. **数据文件**: `tsmc_es_data/{事件名称}.csv`
2. **标签文件**: `tsmc_label.json`（包含新生成的标签）
3. **预测结果**: `tsmc_es_data/{事件名称}/output.json`
4. **可视化图表**: `tsmc_es_data/{事件名称}/output.png`

## 标签生成规则

系统会自动为关键词实体生成两两组合的标签，每个组合包含4种关系：
- `实体A 利好 实体B`
- `实体B 利好 实体A`
- `实体A 不利好 实体B`
- `实体B 不利好 实体A`

例如，对于实体["台积电", "英伟达", "美国"]，将生成以下标签：
- 台积电 利好 英伟达
- 英伟达 利好 台积电
- 台积电 不利好 英伟达
- 英伟达 不利好 台积电
- 台积电 利好 美国
- 美国 利好 台积电
- 台积电 不利好 美国
- 美国 不利好 台积电
- 英伟达 利好 美国
- 美国 利好 英伟达
- 英伟达 不利好 美国
- 美国 不利好 英伟达

## 依赖要求

确保已安装以下Python包：
```bash
pip install elasticsearch matplotlib pymongo
```

## 注意事项

1. **ElasticSearch连接**: 确保ES服务正在运行且可访问
2. **文件权限**: 确保脚本有读写相关目录的权限
3. **依赖包**: 确保所有必需的Python包已安装
4. **模型文件**: 确保`main.py`和相关模型文件存在且可运行

## 错误处理

脚本包含完整的错误处理机制：
- ES连接失败时会提示错误信息
- 数据文件不存在时会检查并提示
- 模型运行失败时会记录错误
- 每个步骤都有状态提示

## 故障排除

### 常见问题

1. **ES连接失败**
   - 检查ES服务是否运行
   - 检查网络连接
   - 检查ES配置

2. **数据文件未生成**
   - 检查ES查询条件
   - 检查索引名称
   - 检查日期范围

3. **模型运行失败**
   - 检查依赖包是否安装
   - 检查模型文件是否存在
   - 检查数据格式是否正确

### 调试模式

如需查看详细输出，可以修改脚本中的`subprocess.run`调用，移除`capture_output=True`参数。 