## 目的：

在电脑上建立简单的HTTP服务器，局域网内共享某个文件夹或文件

## 开启流程：

首先下载Python，命令行输入**python -V**，可以查看到版本信息

执行如下命令：

```shell
# 创建一个最简单的 HTTP 服务器
python -m http.server

# 服务器默认监听端口是 8000，支持自定义端口号
python -m http.server 9000

# 服务器默认绑定到所有接口，可以通过 -b/--bind 指定地址，如本地主机：
python -m http.server --bind 127.0.0.1

# 服务器默认工作目录为当前目录，可通过 -d/--directory 参数指定工作目录：
python -m http.server --directory /tmp/

# 或者cd到目标文件夹，然后输入默认命令，会共享当前文件夹下的文件
cd tmp/
python -m http.server 

```

本机查看文件：打开浏览器输入**localhost:8080**

```shell
# 查看ip命令
ipconfig
```

局域网内其他电脑查看文件目录：打开浏览器输入服务器电脑的ip加端口号，如192.168.2.16:8080，如果时本机直接输入http://localhost:8000/ 即可看到目录。

对于具体文件，要么在目录中打开，要么以地址形式访问 http://localhost:8000/FILENAME

## 可能的问题

如果输出 Serving HTTP on :: port 8000 (http://[::]:8000/) 时需要先访问下 http://localhost:8000/ 才能正常访问