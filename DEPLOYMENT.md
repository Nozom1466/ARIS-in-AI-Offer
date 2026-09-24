# 个人学习版的更新与部署

本站地址：[Nozom1466 / ARIS-in-AI-Offer](https://nozom1466.github.io/ARIS-in-AI-Offer/)。本 fork 保留上游教程、原作者署名和历史审查记录，新增内容由本仓库维护。

## 修改与发布

1. 修改 `docs/tutorials/*.md`。对应中英文页面应同步更新；可运行代码放在 `docs/tutorials/code/`。
2. 新增教程时，把渲染参数加入 `tools/tutorials_render_manifest.json`，并在 `tools/build_index.py` 的 `CATALOG` 注册；新增脚本也在对应教程的 `code` 列表中注册。
3. 本地执行以下命令。提交源文件和生成文件后推送 `main`，GitHub Actions 会再次运行测试、从 Markdown 重建 HTML、校验产物并发布到 Pages。

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python tools/build_site.py
python tools/verify_reviews.py --mode bootstrap --reproduce
```

首次部署需要在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。无需额外配置 PAT 或第三方托管服务的密钥。部署日志在 **Actions → Deploy GitHub Pages**，也可手动运行该 workflow。

`tools/site_config.py` 统一决定首页、教程导航与代码链接的地址。Actions 中默认使用当前 `GITHUB_REPOSITORY`；本地默认是 `Nozom1466/ARIS-in-AI-Offer`。其他 fork 可设置 `ARIS_REPOSITORY`，自定义域名可设置 `ARIS_SITE_URL`。

## 校验与独立审查的区别

本 fork 的默认发布条件是 **RoPE 数值测试 + 文档代码一致性 + 源文件和 HTML 的可复现校验**。这些自动检查不等于跨模型或人工独立审查。

原有 `.review.json` 记录的是当时版本的审查。本次新增 RoPE 没有新的独立跨模型审查，因此不修改旧记录的源码哈希、线程 ID 或 PASS 结论来假装重新审过。源码修改后，校验会显示 `REVIEW_STALE`，作为明确的提醒；HTML 与源码不符、渲染无法复现、已有审查明确失败等仍会阻止发布。`--reproduce` 也会检查这些旧审查对应的当前 HTML。

需要原仓库的严格审查条件时，在 **Source and review audit** 手动运行时勾选 `strict_reviews`，或执行：

```bash
python tools/verify_reviews.py --mode strict --reproduce
```

严格模式会在缺少当前版本独立审查时失败，这是预期行为。上游的 `CONTRIBUTING*.md` 和 `skills/` 保留为原始流程参考；本 fork 的日常发布以本页为准。

## 本次 Attention 补充

- §8.1：RoPE 旋转公式、相邻配对与前后半段配对。
- §8.2：两种简洁 PyTorch 实现、接入 MHA 的位置、KV cache 偏移，以及 checkpoint 布局不可直接互换的原因。
- `tests/test_rope.py`：复数参考对照、布局转换、位置零、范数、dtype、逐 token 位置、相对位移、梯度和中英文代码一致性。

文档公式使用 `$...$` / `$$...$$`，兼容 GitHub 和本站 MathJax 渲染。
