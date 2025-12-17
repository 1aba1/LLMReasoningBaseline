from pathlib import Path


def resolve_path(root_dir: Path, relative: str) -> str:
  """把config.yaml 中的相对路径转成绝对路径。
  这样无论你从哪个工作目录执行 `python main.py`，路径都能正确解析。
  
  参数
  ------
  root_dir: Path
    项目根目录路径
  relative: str
    相对路径字符串
  
  返回
  ------
  str
    绝对路径字符串
  """
  return str((root_dir / relative).resolve())