# 1、写在前面
1
网页地址：https://hankin2015.github.io/Machine_to_DeepingLearning/

使用gitbook+github pages+github actions进行搭建。

最终发现使用docker环境能正常部署，但是会自动创建gh-pages分支发布html文件。

# 2、目录
- ACM

# 方案操作方法
使用 GitBook 结合 GitHub Pages 时，确实可以不需要手动上传编译后的 HTML 文件，通过 GitHub Actions 自动化流程可以实现这一目标。

## 1、核心思路
利用 GitHub Actions 实现「推送源码后自动编译 GitBook 并部署到 GitHub Pages」，全程无需手动处理编译产物（HTML 文件），只需维护 Markdown 源码即可。

## 2、准备 GitBook 源码
在 GitHub 仓库中存放 GitBook 源文件（book.json、SUMMARY.md、Markdown 文档等），无需包含 _book 目录（编译产物），建议在 .gitignore 中添加：
```
docs/_book/
docs/node_modules/
```
由于该项目还有其他文件需要上传，并不是单一的文档书籍，因此需要添加docs文件夹。此方式参考https://github.com/alibaba/async_simple/blob/main/.github/workflows/static.yml

## 3、配置 GitHub Actions 工作流
在仓库中创建自动化部署的工作流配置文件：

仓库根目录新建文件夹 .github/workflows/
在该文件夹下创建 deploy.yml 文件，内容如下：
```
name: GitBook Deploy

on:
  push:
    branches: [ master ]  # 监听 master 分支的推送

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: 拉取代码
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: 安装 Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 16  # GitBook 对 Node 版本有要求，建议 14+

      - name: 安装 GitBook 工具
        run: |
          npm install -g gitbook-cli
          gitbook install  # 安装 book.json 中配置的插件

      - name: 编译 GitBook
        run: 
            cd docs
            gitbook build  # 生成 _book 目录（编译后的 HTML）

      - name: 部署到 GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_book  # 指定编译产物目录
```

## 4、配置 GitHub Pages 源
- 进入 GitHub 仓库 → Settings → Pages
- 在「Source」中选择：GitHub Actions（然后你就会发现github会提供默认的两种配置yml文件）
- 保存配置后，GitHub Pages 会自动创建并读取编译后的 HTML 文件。

## 5、部署报错
```
Run `npm audit` for details.
Installing GitBook 3.2.3
/opt/hostedtoolcache/node/16.20.2/x64/lib/node_modules/gitbook-cli/node_modules/npm/node_modules/graceful-fs/polyfills.js:287
      if (cb) cb.apply(this, arguments)
                 ^

TypeError: cb.apply is not a function
    at /opt/hostedtoolcache/node/16.20.2/x64/lib/node_modules/gitbook-cli/node_modules/npm/node_modules/graceful-fs/polyfills.js:287:18
    at FSReqCallback.oncomplete (node:fs:203:5)
Error: Process completed with exit code 1.
```
这个错误通常是由于 GitBook 与高版本 Node.js 不兼容 导致的。GitBook 官方已停止维护，其依赖的 graceful-fs 等库与 Node.js 14+ 版本存在兼容性问题，具体表现为回调函数处理异常。

GitBook 对 Node.js 版本较为敏感，推荐使用 Node.js 12.x（经测试兼容性最佳）。同时需要手动处理依赖冲突。修改了多个版本解决方法，经测试，未解决。

替代方案（推荐）：使用社区维护版本
由于 GitBook 官方已停止维护，长期来看建议迁移到社区维护的分支（如 gitbook-legacy）或替代工具（如 mdBook、VitePress）。