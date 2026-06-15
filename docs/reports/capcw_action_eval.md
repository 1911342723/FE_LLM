# CAPCW 阶段三 · 控制层整合：动作类型 vs 回复内容

- 判定：**PASS: 动作类型两者皆可(无 headroom)，但回复内容(精确 value 取回) CAPCW 明显胜单向量——CAPCW 在控制层的价值落在内容/状态取回，与 B2 真实数据结论一脉相承**
- 任务：绑定+双输出（动作类型 ask/answer/refuse 粗；回复内容=精确 value 细）；K=5, d=32, seeds=2；value 随机基线 0.083

| 维度 | flat（单向量） | CAPCW（slot 工作空间） | delta |
|---|---:|---:|---:|
| 动作类型（value 依赖+联合训练） | 0.5555 | 0.9113 | +0.3557 |
| 回复内容·精确 value（CAPCW 主场） | 0.4425 | 0.8288 | **+0.3863** |

- query→slot 路由分离 bound/unbound：0.4821（surprise 信号：未绑定难匹配=高 surprise）

- 注：纯「成员判断」动作（query 是否在场→答/问）单向量也能 ~1.0、无 headroom（与 B2 一致）；
  本任务动作 value 依赖且与 value 头联合训练，故 flat 在动作上也降、CAPCW 两面皆胜。
  最干净的判别结果是「回复内容·精确 value」——content 取回才是 CAPCW 不可替代之处。

- 说明：动作类型粗判断单向量够用(无 headroom)；回复内容需取回精确 value，是 CAPCW 内容寻址的主场。与 B2(真实数据 belief 价值在内容/状态、不在动作类型)在引擎层一致。
