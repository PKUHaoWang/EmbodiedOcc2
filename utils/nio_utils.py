import niofs
import io
import os
import numpy as np
import time
import json
from read_cache_op_py import read_cache_op_py

access_key="a0d73ea9b4f6f3a5"
secret_key="ae855ff38cba46428f06f6c150432eb6"
bucket_name='ad-cn-hlidc-gcc-vlm'
client = niofs.Client(access_key, secret_key)
root_dir = "xiaobao.wei/occ/EmbodiedOcc-SDF/"

# 缓存文件的路径
CACHE_FILE = "file_size_cache.jsonl"  # 使用 .jsonl 扩展名表示每行一个 JSON 对象

# 在脚本启动时加载缓存到内存
def load_cache():
    """加载缓存文件"""
    cache = {}
    if not os.path.exists(CACHE_FILE):
        return cache
    with open(CACHE_FILE, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            cache.update(entry)
    return cache

# 全局缓存变量
cache = load_cache()

def save_cache_entry(path, size):
    """将单个缓存条目追加到缓存文件中"""
    entry = {path: size}
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def join_root_dir(path):
    if "./" in path:
        path = path.replace("./", "")
    return os.path.join(root_dir, path)

def download_file_as_mem_file(key):
    key = join_root_dir(key)
    tmp_file = io.BytesIO()
    client.download_fileobj(bucket_name, key, tmp_file)
    tmp_file.seek(0)
    return tmp_file

def download_file_as_bytes(key):
    key = join_root_dir(key)
    tmp_file = io.BytesIO()
    client.download_fileobj(bucket_name, key, tmp_file)
    tmp_file.seek(0)
    return tmp_file.getvalue()

def upload_file(file_path_or_obj, key):
    key = join_root_dir(key)
    breakpoint()
    client.upload_file(file_path_or_obj, bucket_name, key)

def upload_mem_np_file(key, mem):
    key = join_root_dir(key)

    mem_file = io.BytesIO()
    np.save(mem_file, mem)
    mem_file.seek(0)  # 重置文件指针到开始位置

    client.upload_fileobj(mem_file, bucket_name, key)

def upload_mem_ply_file(key, ply_data):
    key = join_root_dir(key)

    mem_file = io.BytesIO()
    ply_data.write(mem_file)
    mem_file.seek(0)  # 重置文件指针到开始位置

    client.upload_fileobj(mem_file, bucket_name, key)

def listdir(prefix):
    prefix = join_root_dir(prefix)
    dirs,files = client.list_dir(bucket_name, prefix)
    dirs.extend(files)
    return dirs

def object_exists(key):
    key = join_root_dir(key)
    return client.object_exist(bucket_name, key)

def down_dir(prefix, path):
    prefix = join_root_dir(prefix)
    return client.download_dir(bucket_name, prefix, path)


def get_ceph_size(path):
    """获取 Ceph 文件大小，优先从缓存中读取"""
    path = join_root_dir(path)

    # 如果缓存中存在，直接返回
    if path in cache:
        return cache[path]

    # 如果缓存中不存在，请求服务器
    try:
        response_info = client.head_object(Bucket=bucket_name, Key=path)
        size = response_info["ContentLength"]
        # 将结果写入缓存
        cache[path] = size
        save_cache_entry(path, size)  # 追加新的缓存条目到文件
        return size
    except Exception as exceptioninfo:  # pylint: disable=broad-except # noqa: B902
        print(exceptioninfo)
        print(path)
        return None


def download_cache_as_mem_file(path):
    if not isinstance(path, list):
        path = [path]
    # s_time = time.time()
    ceph_path_list = ["/" + os.path.join(bucket_name, join_root_dir(p)) + " " + str(get_ceph_size(p)) for p in path]
    # print("get size: ", time.time() - s_time)
    # s_time = time.time()
    raw_bytes = read_cache_op_py(ceph_path_list)[0]
    # print("read cache: ", time.time() - s_time)
    raw_bytes = io.BytesIO(raw_bytes)
    return raw_bytes

def download_cache_as_bytes(path):
    if not isinstance(path, list):
        path = [path]
    ceph_path_list = ["/" + os.path.join(bucket_name, join_root_dir(p)) + " " + str(get_ceph_size(p)) for p in path]

    raw_bytes = read_cache_op_py(ceph_path_list)[0]
    raw_bytes = io.BytesIO(raw_bytes)
    return raw_bytes.getvalue()
