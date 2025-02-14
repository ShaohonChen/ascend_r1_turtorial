#

## 下载模型和数据集

下载数据集

```bash
git clone https://modelers.cn/chenshaohon/Countdown-Task-3to4.git
```

下载模型

```bash
openmind-cli pull zhouhui/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct
```

## 第二步

下载训练好的模型

```bash
openmind-cli pull chenshaohon/qwen-3b-r1-coundown --local_dir ./output/qwen-3b-r1-coundown
```

## 其他脚本

转换notebook脚本

```bash
pip install nbconvert
jupyter nbconvert [OPTIONS] notebook.ipynb
# --to <format>：指定输出格式。可以选择多种格式，包括：
#     html：将 notebook 转换为 HTML 格式。
#     pdf：将 notebook 转换为 PDF 格式。
#     python：将 notebook 中的代码导出为 Python 脚本。
#     markdown：将 notebook 转换为 Markdown 格式。
#     notebook：将 notebook 转换为 .ipynb 格式。
#     latex：将 notebook 转换为 LaTeX 格式。
# --execute：在转换过程中执行 notebook 中的代码。如果不指定，默认只会转换文本和代码，不会执行代码。
# --inplace：直接修改原始的 notebook 文件。如果不指定，nbconvert 会创建一个新的输出文件。
# --output <filename>：指定输出文件的路径和名称。如果没有指定，nbconvert 会自动生成一个文件名。
# --ExecutePreprocessor.kernel_name=<kernel>：指定使用的内核（如 python3），适用于 notebook 使用不同语言的场景。
# --allow-errors：允许在执行 notebook 时忽略错误并继续执行。如果 notebook 中的代码单元出现错误，默认情况下执行会停止，添加此选项可以忽略错误。
```

