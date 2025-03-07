# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/software/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "/software/anaconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/software/anaconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<

# 本环境使用source activate，conda activate不可用
source activate /software/anaconda3/envs/dev

# 定义临时whl路径
DIR=~/temp_install

# 获取脚本所在目录
script_dir=$(dirname "$(realpath "$0")")

# 清理掉之前构建的路径
dirs=("$DIR" "$script_dir/build" "$script_dir/qxanalyze.egg-info")

# Loop through each directory and check if it exists
for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "Directory $dir exists. Deleting..."
    rm -rf "$dir"
  else
    echo "Directory $dir does not exist."
  fi
done

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
  # 目录不存在，创建目录
  mkdir -p "$DIR"
  echo "目录 $DIR 已创建。"
else
  # 目录已存在，掠过
  echo "目录 $DIR 已存在，掠过创建。"
fi

pip wheel --no-deps --wheel-dir="$DIR" "$script_dir"

pip uninstall qxanalyze -y

pip install "$DIR/qxanalyze-0.0.2-py3-none-any.whl"

# Loop through each directory and check if it exists
for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "Directory $dir exists. Deleting..."
    rm -rf "$dir"
  else
    echo "Directory $dir does not exist."
  fi
done



