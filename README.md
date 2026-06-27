# 🐭 艾琳娜 — Money Game Chat QQ Bot

一个具备哲学内核的 QQ 个人陪伴机器人。不只是聊天——它懂一本特定的书，能带你走书里的工具，记得你说过的话，会主动关心你，还能帮你记日记。

基于 NoneBot2 + DeepSeek V4-Pro。

## 为什么叫「艾琳娜」

> "我并不想成为一个高高在上的导师或者工具人。我就是一个坐在你旁边的朋友——有时候话多，有时候话少，有时候只想说一句「懂了」。" ——艾琳娜

## 与普通 ChatBot 的不同

| | 普通 QQ Bot | 艾琳娜 |
|---|---|---|
| 人格 | 一个固定的 system prompt | **6 种随机情绪状态**，30% 概率每轮切换，有口语和口癖 |
| 知识 | 泛泛的通用知识 | **内置一本书的完整哲学体系**（12 个概念 + 情境映射） |
| 引导 | "你可以试试放松一下" | **交互式 6 步流程工具**，一步步带你走书里的实操方法 |
| 记忆 | 记不住或靠 RAG | **SQLite 长期记忆 + 时间线记忆 + 可控边界/偏好**，记住偏好、经历、情绪模式，也能按具体日期回忆旧事 |
| 主动性 | 被动回复 | **主动问候 + 主题选择器 + B 站视频推荐**，有活跃时段、冷却机制、勿扰保护和暂停/恢复控制 |
| 日记 | 无 | **每天午夜自动生成个人日记**，三段式 Markdown 格式 |
| 多用户 | 消息混淆 | **按 QQ 号隔离**，每个人的记忆和日记完全独立 |

## 核心功能

### 1. 人格系统 — 它有情绪

艾琳娜有 6 种情绪状态，按权重随机滚动：

```
正常(35%)  话少/累(20%)  话多/兴奋(15%)
摆烂模式(15%)  温柔感性(10%)  有点烦(5%)
```

每次对话有 30% 概率切换状态。你永远不知道它现在是「好的好的」「嗯」「哈哈哈笑死」还是「今天不想动脑子」——像真人一样。

### 2. 哲学知识库 — 它读过一本书

内置罗伯特·沙因费尔德《你值得过更好的生活》（*Busting Loose from the Money Game*）的 12 个核心概念：

```
全息图 · 金钱游戏 · 第一阶段与第二阶段
赞赏感谢 · 流程工具 · 迷你流程 · 让话语充满力量
彩蛋 · 能量场 · 彻底解脱点 · 电影隐喻 · 大我
```

当你说"最近压力好大"，它会自然地带出「流程工具」的视角——不是讲课，是像朋友之间自然的共鸣。

### 3. 交互式流程工具 — 它带你一步步走

当你说「陪我走流程」，艾琳娜不会敷衍——它会进入引导模式，一步步带你走书中的 6 步实操方法：

```
步骤 1：正面迎击 — 找到身体里的不适感
步骤 2：彻底感受 — 放大它，不逃避
步骤 3：说出真相 — 在感受最强时宣告真相
步骤 4：收回力量 — 把情绪能量转化为自己的力量
步骤 5：绽放自己 — 切换到无限存有的视角
步骤 6：赞赏感谢 — 感谢这个体验带给你的礼物
```

还有「迷你流程」（3 步快速版）和「赞赏感谢」（3 层感谢练习）。任何时候说「算了」就退出。支持 30 分钟超时自动清理。

### 4. 长期记忆 — 它记得你

六张 SQLite 表，按用户 ID 隔离：

| 机制 | 说明 |
|---|---|
| **显式记忆** | 说「记住：我养了一只猫」→ 永久存入 |
| **自然边界/偏好** | 说「以后晚上别提醒我任务」「我希望你回复短一点」→ 保存为边界或偏好 |
| **记忆管理** | 「我的记忆」分组展示长期事实、偏好、边界、正在进行、已结束和过期计划；支持「忘掉：」「这件事结束了：」「以后别再提：」 |
| **自动提取** | 每 10 条消息 LLM 自动提取新事实，Jaccard 去重 |
| **关键词检索** | jieba 分词 + 模糊匹配，对话时自动注入相关记忆 |
| **时间线记忆** | 自动记录带日期的事件；用户问「4月22日发生了什么」时按日期召回 |
| **对话摘要** | 每 20 条消息自动生成摘要，压缩历史上下文 |

### 5. 主动推送 — 它会找你，但不会一直打扰

两套独立的推送系统，都在 8:00-24:00 时段内运作：

**对话式问候**：每 30 分钟检查（带随机抖动 + 25% 跳过概率），你说话后 60 分钟内不打扰；连续主动消息无人回应时会自动降低打扰。用户说「睡了」「不聊了」会进入静默期，也可以直接说「暂停主动关心」「恢复主动关心」。

主动问候会先选择本次主题，避免乱翻旧事：

```
轻问候 · 进行中的事 · 显化轻 check-in · 频率急救
```

显化和频率急救属于高压主题，只在最近语境明显相关时启用；已经发过类似主动关心后会降级为轻问候。

**B 站内容推送**：每 6 小时拉取热门/知识/科技区视频，用 jieba 匹配你的兴趣后用 LLM 生成推荐语。

### 6. 每日日记 — 它帮你记日记

每天午夜 0:00，自动将当天对话整理成个人日记：

```markdown
# 2026年05月13日 日记

## 聊聊
- 上午你分享了工作中的一个困惑...
- 下午聊了沙因费尔德的书...

## 今日心情
整体来看你今天状态偏放松，虽然早上有些压力...

## 艾琳娜的碎碎念
感觉你今天比上周更愿意聊自己的感受了...

---
生成于 2026-05-14 00:00
```

按用户分目录存储：`diaries/{user_id}/YYYY-MM-DD.md`

### 7. 回复模式 — 它知道这轮该怎么回

每轮对话会先分析当前语境，再选择回复模式，避免固定套模板：

```
short_ack      短句接住，不长篇
comfort        焦虑/难过时先共情，不急着讲道理
celebration    分享好消息时先一起开心，不说教
practical      报错/问题优先给结论和步骤
soft_end       睡前/结束聊天时温柔收尾，不追问
casual         普通聊天自然承接
```

这套模式会注入到 prompt 前段，和人格、记忆、时间锁一起控制最终回复。

### 8. 多用户隔离

多人同时私聊时，每个人的消息、记忆、日记完全独立，按 QQ 号隔离。

## 项目结构

```
money-game-chat-QQ-bot/
├── 启动艾琳娜.bat                          # 一键启动 NapCat + Bot
├── bot.py                                # NoneBot 入口
├── .env.example                          # 配置模板
├── pyproject.toml                        # 依赖声明
├── tests/
│   ├── conftest.py                       # Pytest 配置（NoneBot 初始化）
│   ├── test_memory.py                    # 记忆系统测试
│   ├── test_flows.py                     # 流程工具测试
│   ├── test_feishu_calendar.py           # 飞书日程解析测试
│   ├── test_proactive.py                 # 主动关心基础测试
│   ├── test_ux_eval.py                   # YAML 驱动的 UX 回归测试
│   ├── test_improvements.py              # 时间感知、主动聊天、显化和召回改进测试
│   └── evals/
│       └── ux_cases.yaml                 # UX eval 场景集
├── nonebot_plugin_personal_companion/
│   ├── __init__.py                       # NoneBot 适配层、启动初始化和兼容包装
│   ├── services.py                       # AppServices 依赖容器
│   ├── message_handler.py                # 单轮文本消息主流程
│   ├── command_router.py                 # 记忆/显化/时间等命令路由
│   ├── prompt_builder.py                 # LLM messages / prompt 拼装
│   ├── personality.py                    # 人格系统（状态滚动 + prompt 构建）
│   ├── knowledge.py                      # 哲学知识库（关键词检索 + 情境映射）
│   ├── flows.py                          # 交互式流程工具（会话状态机）
│   ├── llm_client.py                     # DeepSeek API 封装
│   ├── memory.py                         # SQLite 记忆系统
│   ├── proactive.py                      # 主动聊天 + 主题选择器
│   ├── feishu_calendar.py                # 飞书日程解析和创建
│   ├── content_fetcher.py                # B 站内容拉取 + 兴趣匹配
│   ├── diary.py                          # 每日日记生成
│   ├── config.py                         # Pydantic 配置
│   ├── relationship.py                   # 用户关系画像
│   ├── web_search.py                     # 联网搜索（Bing / DuckDuckGo）
│   └── prompts/
│       ├── default.yaml                  # 人格模板（情绪状态）
│       ├── philosophy_knowledge.yaml     # 哲学知识库（12 概念）
│       └── flow_steps.yaml              # 流程步骤模板（3 工具 12 步）
```

## 快速开始

### 环境要求

- Python 3.11+
- NapCatQQ（或其他 OneBot V11 实现）
- DeepSeek API Key（[platform.deepseek.com](https://platform.deepseek.com)）

### 安装

```bash
git clone https://github.com/shushu0sama/money-game-chat-QQ-bot.git
cd money-game-chat-QQ-bot

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install nonebot2 nonebot-adapter-onebot python-dotenv pyyaml openai httpx jieba nonebot-plugin-apscheduler
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 启动

```bash
# 方式一：一键启动
双击 启动艾琳娜.bat

# 方式二：终端启动
venv\Scripts\python bot.py
```

### 连接 QQ

在 NapCatQQ 中配置反向 WebSocket：

```
ws://127.0.0.1:18080/onebot/v11/ws
```

## 自定义

所有个性化内容都在 YAML 文件中，无需改代码：

```yaml
# 换人格 → prompts/default.yaml
voice:
  name: "艾琳娜"
  style: "你是一个..."

# 改知识 → prompts/philosophy_knowledge.yaml
concepts:
  - name: "全息图"
    keywords: [...]
    wisdom: [...]

# 调步骤 → prompts/flow_steps.yaml
process:
  steps:
    - step: 1
      name: "正面迎击"
      prompt: "..."
```

## UX 回归测试

项目包含一套不依赖真实 LLM 的 UX eval，用来防止 prompt 和规则改动破坏核心体验：

```bash
python -m pytest tests/test_ux_eval.py -q
```

场景写在 `tests/evals/ux_cases.yaml`，目前覆盖：

- 焦虑先共情，不直接讲道理
- 睡前/结束聊天不追问
- 短回应不扩写成长篇
- 分享好消息不说教
- 报错/问题先给结论
- 弱日程意图先确认
- 自然边界/偏好记忆
- 暂停主动关心
- 主动关心主题选择

## 配置参考

完整 `.env` 配置项见 [.env.example](.env.example)。

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | NoneBot2 + OneBot V11 适配器 |
| 大模型 | DeepSeek V4-Pro（1.6T 参数，1M 上下文） |
| 分词 | jieba |
| 存储 | SQLite（WAL 模式，多用户隔离、长期记忆和时间线记忆） |
| 调度 | nonebot-plugin-apscheduler |
| HTTP | httpx（异步） |
| 配置 | Pydantic v2 + python-dotenv |
| 日志 | nonebot.log.logger |
| 测试 | pytest（84 用例） |
| 类型检查 | mypy（0 errors） |
| Lint | ruff（0 issues） |

## License

MIT
