import subprocess
import sys
import os

def run_commands_sequentially(program, base_args, arg_list, shell_type="powershell", verbose=True):
    """
    顺序执行一个程序多次，每次使用不同的命令行参数，并确保使用指定的Shell环境。

    :param program: 要运行的程序路径（字符串）。
    :param base_args: 程序的基础参数（列表）。
    :param arg_list: 不同的参数列表，每个元素是一个参数列表（列表的列表）。
    :param shell_type: 使用的Shell类型，如 "powershell" 或 "cmd"。
    :param verbose: 是否打印详细的运行信息（布尔值）。
    """
    # 获取当前环境变量
    env = os.environ.copy()

    for args in arg_list:
        # 构建完整的命令字符串，包括 base_args 和 args
        # args用列表会通用一些
        full_args = base_args + args#其实传字符串就足够了
        cmd = f"{sys.executable} {program} " + " ".join(full_args)
        if verbose:
            print(f"运行命令: {cmd}")

        try:
            # 根据Shell类型构建subprocess.run的参数
            if shell_type.lower() == "powershell":
                # PowerShell中需要使用 -Command 参数，并确保引号正确
                full_cmd = ["powershell", "-Command", cmd]
            elif shell_type.lower() == "cmd":
                # CMD中直接传递命令字符串
                full_cmd = cmd
            else:
                # 默认不使用Shell，直接传递列表形式的命令
                full_cmd = [sys.executable, program] + base_args + args

            # 执行命令并等待完成
            result = subprocess.run(
                full_cmd,
                shell=True if shell_type.lower() in ["powershell", "cmd"] else False,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                env=env
            )
            if result.returncode != 0:
                print(f"命令执行失败: {' '.join(full_cmd)}", file=sys.stderr)
                print(f"错误代码: {result.returncode}", file=sys.stderr)
                # 根据需求决定是否继续执行下一个命令
                # 如果希望在失败时停止，可以取消下面的注释
                # sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {' '.join(e.cmd)}", file=sys.stderr)
            print(f"错误代码: {e.returncode}", file=sys.stderr)
            print(f"错误输出: {e.stderr}", file=sys.stderr)
            # 根据需求决定是否继续执行下一个命令
            # 如果希望在失败时停止，可以取消下面的注释
            # sys.exit(1)

def main():
    # 要运行的程序（假设在当前目录下）
    program = "2d_pos_encoding/main.py"

    # 基础参数（如果有）
    base_args = ["--L 10 -pd 2d_pos_encoding/src/1.jpg  -v --epochs 20","-reg"]

    # 不同的参数列表
    arg_list = [
        ["0"],["0.1"],["10"],["100"],["1"]
        # 可以继续添加更多参数
    ]

    # 检查程序是否存在
    if not os.path.isfile(program):
        print(f"程序 {program} 不存在。请确保程序路径正确。", file=sys.stderr)
        sys.exit(1)

    # 运行命令，指定使用的Shell类型
    run_commands_sequentially(program, base_args, arg_list, shell_type="powershell")

if __name__ == "__main__":
    main()
