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
    branches: [ main ]  # 监听 main 分支的推送

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