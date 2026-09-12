# 文档站维护

网站由 Docsify 直接读取仓库中的 Markdown，不需要 Node 构建或复制正文。
教程在 `docs/tutorials/`，操作指南仍在 `docs/`。

## 本地预览

从仓库根目录运行：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

浏览器打开 `http://127.0.0.1:8000/`。
如果 8000 已被占用，换一个空闲端口；不要关闭其他项目的服务。
停止预览时在这个终端按 Ctrl+C。

不能双击 `index.html` 使用 `file://` 阅读：Docsify 需要通过 HTTP 请求 Markdown。
首次加载需要访问 jsDelivr CDN；离线时仍可直接阅读本地 Markdown。

## 发布到 GitHub Pages

仓库管理员首次配置：

1. 将文档站文件提交并推送到准备发布的分支。
2. 打开仓库 Settings → Pages，Source 选择 **Deploy from a branch**。
3. 选择发布分支（通常为 `main`）以及 **`/docs`** 目录，保存。
4. 等待 GitHub Pages 部署完成，打开设置页给出的 URL。

`docs/.nojekyll` 保留 `_sidebar.md` 等以下划线开头的文件。
站内使用 hash 路由，所以项目子路径部署和直接打开章节链接都不需要服务器重写。
本次只提供文件和配置步骤，没有替你推送代码或启用远端 Pages。

## 新增一篇文章

1. 在 `docs/tutorials/` 中添加 Markdown，文件名沿用 `03-...md` 的编号风格。
2. 在 `docs/_sidebar.md` 中把对应“待写”项换成链接。
3. 更新 `docs/tutorials/README.md` 的状态和相邻文章的前后篇链接。
4. 本地预览正文、手机目录、搜索和源代码链接，重新运行文中的实验。

搜索和站内章节翻页读取侧栏中的已完成章节，不需要维护另一份 JavaScript 章节清单。
搜索索引在浏览器中缓存十分钟，新增章节后可清除站点存储或稍后刷新。
未完成章节保留普通文字，不建立空页面。

## 公式、图解与查阅区

公式使用 `math` 代码围栏，内容为 LaTeX。
图解使用 `mermaid` 代码围栏，尽量使用纵向结构，并写明 `accTitle` 和 `accDescr`，
使手机端和辅助技术也能理解图意。数学公式和 Mermaid 也能在 GitHub Markdown 中阅读。

完整命令、校验代码可以放入 `<details>` 和 `<summary>` 查阅区。
HTML 标签与内部 Markdown 之间保留空行，确保 GitHub 与 Docsify 都能解析。
正文先解释问题和例子，真正影响结果的边界不要全藏到折叠区。

相对 Markdown 链接继续按文件所在目录写。
指向 `docs/` 内的文档在站内跳转；指向 `src/`、`scripts/`、`configs/` 等仓库文件时，
阅读站打开 GitHub 对应源码。原始 Markdown 中的链接仍然可以使用。

## 固定依赖

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| Docsify | 4.13.1 | Markdown、路由和官方搜索插件 |
| KaTeX | 0.16.11 | 数学公式 |
| Mermaid | 10.9.3 | 数据流和结构图 |
| Prism | 1.29.0 | 补充 Bash、Python、YAML、JSON 高亮 |
| Lucide | 0.468.0 | 导航和工具图标 |

依赖在 `docs/index.html` 中通过固定版本 CDN URL 加载，
没有使用 `latest`。它们不是 Python 训练依赖，也不影响模型实验环境。
更新版本后应重新验证搜索、公式、图解与移动端菜单。

## 浏览器检查

站点启动后，可以运行独立的 Playwright 冒烟检查：

```bash
npm install --prefix build/docs-tools --no-save playwright@1.58.2
build/docs-tools/node_modules/.bin/playwright install chromium
NODE_PATH=build/docs-tools/node_modules node tests/docs/site-smoke.cjs http://127.0.0.1:8000
```

仅需要测试文档站时安装这套工具；训练项目不需要 Node。
截图写入已被 Git 忽略的 `build/docs-screenshots/`，不混入文章目录。
