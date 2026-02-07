# PRD: Mocha-TIMIT 下载与可执行接入规范（AI执行版）

> 本文件是执行合同，内容与 `/.sisyphus/plans/mocha-timit-prd-todo.md` 保持一致，放在 `docs/` 便于后续 AI 代理直接读取。

## 0. 文档定位

- 目标：给后续 AI 执行代理一份“可直接照做”的任务合同文档。
- 约束：本 PRD 聚焦 **Phase 1（立即可做）**，并定义 **Phase 2（可选延伸）** 的激活门槛。
- 关键原则：零人工介入验证、幂等、可审计、不破坏 ACP v0 合同。

---

## 1. 背景与目标

### 1.1 已有状态

- 已下载（存在于当前 workspace）：`PHOIBLE`、`WikiHan`、`WikiPron`
- 已下载（但当前 workspace 未发现目录）：`CLTS`
- 未下载：`Mocha-TIMIT`
- 现有系统：ACP-first PGDN v0 已可运行，且 confidence/ranking 语义已固定。

### 1.1.1 当前数据集目录结构（已整理）

> 目标：让后续 AI 代理不需要猜路径，直接从统一入口拿数据。

- 统一入口目录：`data/external/`
- 统一链接目录：`data/external/links/`
  - 这里放的是 **symlink**，指向真实数据集位置（避免移动/复制大目录）
  - 例：`data/external/links/phoible -> ../../../phoible`
- 机器可读清单：`data/external/dataset_registry.json`
  - 由脚本生成，记录 present/missing、大小、文件数、link_path
- 刷新脚本：`python3 scripts/organize_datasets.py`

当前扫描结果（以 `data/external/dataset_registry.json` 为准）：

- `PHOIBLE`: present
- `WikiHan`: present
- `WikiPron`: present
- `CLTS`: missing（需要把 CLTS 放进 workspace 或给出路径）
- `Mocha-TIMIT`: missing（本 PRD Phase 1 负责下载到标准位置）

### 1.2 本次目标（Phase 1）

- 建立 Mocha-TIMIT 的标准化下载脚本与证据产物。
- 生成可追溯清单（manifest、校验和、许可证副本、提取目录索引）。
- 输出“如何接入”的执行说明文档，但 **不修改训练/推理合同语义**。

### 1.3 非目标（本次不做）

- 不把 Mocha 数据直接并入 `data/targets/acp_targets.jsonl`
- 不改 `src/pgdn/confidence.py` 的 bucket 逻辑
- 不改 `src/pgdn/infer.py` 输出字段语义
- 不做完整 EMA->ACP 特征工程落地（仅给后续入口）

---

## 2. 范围定义（In/Out）

| Phase | In Scope | Out of Scope |
|------|----------|--------------|
| Phase 1 | 下载、验签、解压、manifest、license 合规标记、执行说明文档 | 模型结构改造、训练数据并表、新的 loss 设计 |
| Phase 2（可选） | 设计 Mocha sidecar 适配层（不破坏 ACP 合同） | 直接替换 ACP 主监督目标 |

---

## 3. 数据源与合规

### 3.1 官方来源

- 语料说明页：`https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html`
- 下载目录：`https://data.cstr.ed.ac.uk/mocha/`

### 3.2 默认下载白名单（必须锁定）

- `mocha-timit_fsew0.tgz` 或等价 v1.1 命名文件
- `mocha-timit_msak0.tgz` 或等价 v1.1 命名文件
- `LICENCE.txt`
- `README_v1.2.txt`（若存在）

### 3.3 合规声明（必须写入 manifest）

- `allowed_use`: research/education/non-commercial
- `commercial_use`: prohibited_without_permission
- `license_source_url`: CSTR 对应链接

---

## 4. 目标产物（Deliverables）

- `scripts/acquire_mocha_timit.sh`
- `data/external/mocha_timit/raw/*`（原始压缩包与 license/readme）
- `data/external/mocha_timit/v1_1/{fsew0,msak0}/...`（解压内容）
- `data/external/mocha_timit/checksums.sha256`
- `data/external/mocha_timit/manifest.json`
- `data/external/mocha_timit/evidence_index.json`
- `docs/mocha_timit_integration.md`（执行说明，强调 Phase 1/2 边界）
- `data/external/links/mocha_timit`（symlink，运行整理脚本后自动生成）

---

## 5. 执行前置条件（Preflight）

- 网络可访问 `data.cstr.ed.ac.uk`
- 本地具备：`bash`、`curl`、`tar`、`python3`、`sha256sum`（或 `shasum -a 256`）
- 目标目录可写：`data/external/mocha_timit`

---

## 6. TODO List（执行清单）

> 规则：每条 TODO 都必须可由 AI 代理独立执行并自动验收。

- [ ] T1. 创建下载脚本骨架 `scripts/acquire_mocha_timit.sh`
  - 输入参数：
    - `--root`（默认 `data/external/mocha_timit`）
    - `--speakers`（默认 `fsew0,msak0`）
    - `--dry-run`（默认关闭）
    - `--force`（默认关闭）
  - 脚本要求：
    - 严格模式：`set -euo pipefail`
    - 仅允许白名单文件下载
    - 不可写 ACP 合同文件

- [ ] T2. 实现下载与重试策略
  - 每个文件使用 `curl -fL --retry 3 --retry-delay 2`
  - 文件已存在且 hash 一致则跳过
  - 下载失败必须返回非零退出码

- [ ] T3. 实现解压与目录规范
  - 原始文件统一进 `raw/`
  - 解压至 `v1_1/{speaker}/`
  - 默认不覆盖已解压目录；仅 `--force` 允许覆盖

- [ ] T4. 生成完整校验和文件
  - 产出：`checksums.sha256`
  - 覆盖范围：`raw/` 下所有下载文件
  - 格式可被 `sha256sum -c` 直接验证

- [ ] T5. 复制并固化许可证证据
  - 复制 `LICENCE.txt` 到 `license/LICENCE.txt`
  - manifest 中写入 license 关键字段（allowed/commercial/source）

- [ ] T6. 生成机器可读 manifest
  - 必含字段：
    - `dataset_name`, `source_urls`, `retrieved_at_utc`
    - `artifacts[]`（文件名/大小/hash）
    - `license`（allowed_use/commercial_use/source_url）
    - `run`（idempotent 布尔值）
    - `integration`（`acp_contract_unchanged: true`）

- [ ] T7. 生成 evidence 索引
  - 文件：`evidence_index.json`
  - 列出：manifest、checksums、license、提取目录清单、命令执行结果

- [ ] T8. 编写接入说明文档 `docs/mocha_timit_integration.md`
  - 内容必须包含：
    - 一键命令
    - 目录结构说明
    - 合规注意事项（非商用）
    - Phase 2 激活门槛
    - 明确声明本阶段不改 ACP 主监督/置信度语义

- [ ] T9. 执行自动验收（首轮）
  - 运行下载脚本并记录退出码
  - 校验 manifest JSON 合法
  - 校验 checksum 全部通过
  - 校验关键目录存在

- [ ] T10. 执行自动验收（二次幂等）
  - 相同命令再运行一次
  - 断言 `run.idempotent == true`
  - 断言无新增重复目录、无重复下载

- [ ] T11. 合同防回归检查
  - 断言未触碰以下文件：
    - `src/pgdn/confidence.py`
    - `src/pgdn/infer.py`
    - `data/targets/acp_targets.jsonl`

- [ ] T12. 输出执行摘要
  - 输出：下载文件总数、总字节数、hash 通过率、是否幂等、license 状态

- [ ] T13. 刷新统一数据集入口（必须做）
  - 运行：`python3 scripts/organize_datasets.py`
  - 断言：`data/external/links/mocha_timit` 存在且指向 `data/external/mocha_timit`

---

## 7. 验收命令（Agent-Executable）

```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
test -f data/external/mocha_timit/manifest.json
test -f data/external/mocha_timit/checksums.sha256
test -f data/external/mocha_timit/license/LICENCE.txt
python3 scripts/organize_datasets.py
test -L data/external/links/mocha_timit
```

```bash
python3 - <<'PY'
import json
p='data/external/mocha_timit/manifest.json'
m=json.load(open(p,'r',encoding='utf-8'))
assert m['dataset_name']=='MOCHA-TIMIT'
assert m['integration']['acp_contract_unchanged'] is True
assert 'commercial' in m['license']['commercial_use'].lower() or 'permission' in m['license']['commercial_use'].lower()
print('manifest-check:ok')
PY
```

```bash
sha256sum -c data/external/mocha_timit/checksums.sha256
```

```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
python3 - <<'PY'
import json
m=json.load(open('data/external/mocha_timit/manifest.json','r',encoding='utf-8'))
assert m['run']['idempotent'] is True
print('idempotent:ok')
PY
```

---

## 8. 风险与回滚

- 风险：CSTR 命名变化导致下载 404
  - 处理：脚本允许候选文件名映射（但仍在白名单内）
- 风险：license 字段缺失导致合规不可审计
  - 处理：manifest 生成失败即整体失败
- 风险：AI 执行时越界修改 ACP 合同文件
  - 处理：加入合同防回归检查，失败即终止

回滚策略：

- 仅删除 `data/external/mocha_timit` 目录即可回到执行前状态
- 不涉及训练/推理主产物回滚

---

## 9. Phase 2（可选）激活条件

仅在以下全部满足时开启：

- Phase 1 全部 TODO 完成且验收通过
- 你明确批准“开始接入 sidecar 适配层”
- 明确不修改 ACP 主监督合同的前提仍成立

Phase 2 可做内容（不在本次执行）：

- 设计 EMA -> 约束特征 sidecar 转换器
- 在 `data/anchors/` 增加生理约束锚点草案
- 增加可选分析脚本，不进入主训练路径

---

## 10. 对 AI 执行代理的硬性指令

- 必须按 TODO 序号执行，不得跳步。
- 每完成一项立即更新状态与证据。
- 禁止将“人工目视检查”作为验收条件。
- 若下载站点异常，必须先产出失败证据，再停止执行。
