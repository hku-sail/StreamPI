# StreamPI 多人开发分支流程

本文档说明如何在不影响 `main` 稳定代码的前提下，创建 `dev` 分支并支持多人并行开发。

## 分支约定

```text
main     稳定分支，只放已经验证过的代码
dev      集成开发分支，多人功能分支先合并到这里
feat/*   新功能分支
fix/*    Bug 修复分支
exp/*    实验性分支
```

推荐协作流程：

```text
feat/* -> dev -> main
```

不要直接在 `main` 上开发新功能。`main` 应该始终保持可运行、可复现。

## 第一次创建 dev 分支

在终端执行：

```bash
cd /Users/liuzhe/Documents/zliu_projs/streamPI
```

进入本地 `streamPI` 仓库目录。后面的 Git 命令都需要在这个目录里执行。

```bash
git status
```

查看当前仓库状态，包括当前所在分支、是否有未提交改动，以及本地分支是否和远端同步。执行分支操作前，建议确认工作区是干净的。

```bash
git fetch origin
```

从远端仓库 `origin` 拉取最新的分支和提交信息。这个命令不会修改当前代码，只会更新本地对远端仓库状态的记录，例如 `origin/main`、`origin/dev` 或其他人的远端分支。

```bash
git switch main
```

切换到 `main` 分支。`main` 通常作为稳定分支，不直接用于日常功能开发。

```bash
git pull origin main
```

把远端 `origin/main` 的最新代码同步到本地 `main`。这样可以保证后面创建的 `dev` 分支是基于最新稳定代码。

```bash
git switch -c dev
```

从当前分支创建一个新的本地分支 `dev`，并立即切换到 `dev`。因为前面已经切到最新的 `main`，所以这里等价于以当前 `main` 为基础创建 `dev`。

```bash
git push -u origin dev
```

把本地 `dev` 分支推送到 GitHub 远端仓库，并把本地 `dev` 与远端 `origin/dev` 绑定。之后在 `dev` 分支上可以直接使用 `git push` 和 `git pull`，不需要每次手动写 `origin dev`。

## 每个人开发新功能

多人协作时，每个人都应该从最新的 `dev` 分支创建自己的功能分支。

```bash
git switch dev
git pull origin dev
```

切换到 `dev` 分支，并同步远端最新的集成代码。

```bash
git switch -c feat/your-feature-name
```

从 `dev` 创建自己的功能分支。`your-feature-name` 应该换成简短、具体、英文小写、用连字符分隔的名称。

示例：

```bash
git switch -c feat/online-continual-learning
git switch -c feat/pi05-online-lora
git switch -c feat/episode-logger
git switch -c feat/replay-buffer
git switch -c feat/eval-gate
git switch -c feat/policy-blue-green-switch
git switch -c fix/checkpoint-saving
git switch -c exp/test-time-adaptation
```

开发完成后提交：

```bash
git add .
git commit -m "add online continual learning workflow"
git push -u origin feat/online-continual-learning
```

然后在 GitHub 上创建 Pull Request：

```text
feat/online-continual-learning -> dev
```

代码 review、测试通过后，再合并到 `dev`。

## dev 合并到 main

当 `dev` 分支已经测试稳定，需要发布到稳定分支时，在 GitHub 上创建 Pull Request：

```text
dev -> main
```

建议只有通过测试、review 和必要的真机或仿真验证后，才把 `dev` 合并到 `main`。

## 推荐保护规则

建议在 GitHub 仓库中设置分支保护：

```text
Settings -> Branches -> Branch protection rules
```

对 `main` 建议开启：

```text
Require a pull request before merging
Require approvals
Require status checks to pass
Restrict force pushes
Restrict deletions
```

对 `dev` 可以开启较轻的保护规则，例如要求 Pull Request 和 CI 通过后才能合并。

## 常用检查命令

查看当前分支和工作区状态：

```bash
git status
```

查看所有本地分支：

```bash
git branch
```

查看本地和远端分支：

```bash
git branch -a
```

查看最近提交：

```bash
git log --oneline --decorate --max-count=10
```

## 总结

推荐长期使用下面的协作方式：

```text
main 保持稳定
dev 作为多人集成开发分支
每个人从 dev 创建 feat/* 或 fix/* 分支
功能完成后通过 PR 合并到 dev
dev 稳定后通过 PR 合并到 main
```
