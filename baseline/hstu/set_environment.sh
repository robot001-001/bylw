#!/bin/bash

#################################################################
# 创建baseline环境


ENV_NAME="hstu_baseline"
PYTHON_VERSION="3.10"




if conda env list | grep -q "^$ENV_NAME "; then
    echo "✅ 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    echo "⚠️ 环境 '$ENV_NAME' 不存在。"
    echo "🚀 正在创建环境: $ENV_NAME (Python $PYTHON_VERSION)..."
    
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    if [ $? -eq 0 ]; then
        echo "🎉 环境 '$ENV_NAME' 创建成功！"
    else
        echo "❌ 创建失败，请检查上方错误日志。"
        exit 1
    fi
fi
echo "------------------------------------------------"
echo "💡 提示: 要激活该环境，请在终端运行:"
echo "conda activate $ENV_NAME"
echo "------------------------------------------------"


