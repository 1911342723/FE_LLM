# -*- coding: utf-8 -*-
"""
data/seed_knowledge.py —— 出厂世界观种子知识库
================================================
对应文档第一步：定义灌入 PostgreSQL 的「出厂世界观」。

设计原则（呼应文档最后的提问：先灌数理骨架还是常识图谱？）：
    采取折中——既放绝对理性的数理/物理公理(depth 极高、几乎不可撼动)，
    也放一批基础常识(depth 适中、深度角色扮演时可适度妥协)。

每条知识是一个 dict：name / text / category / depth / relations。
新增知识只需在此追加，再运行 scripts/seed_db.py 灌库。
"""

from __future__ import annotations

# 分类常量
AXIOM = "公理"        # 数理/物理公理：depth 高，守护逻辑严密性
COMMON = "常识"       # 基础常识：depth 中
DOMAIN = "领域"       # 领域知识：depth 中低


SEED_CONCEPTS: list[dict] = [
    # ============ 一、数理与逻辑公理（depth 高，不可撼动）============
    {
        "name": "序关系传递性",
        "text": "若 A 大于 B 并且 B 大于 C 那么 A 必然大于 C 这是序关系的传递性公理",
        "category": AXIOM, "depth": 3.0,
        "relations": {"互斥": ["循环不等式谬误"]},
    },
    {
        "name": "循环不等式谬误",
        "text": "A 大于 B 且 B 大于 C 同时 C 又大于 A 这是逻辑上不可能成立的循环矛盾",
        "category": AXIOM, "depth": 0.5,
        "relations": {"互斥": ["序关系传递性"]},
    },
    {
        "name": "排中律",
        "text": "任何命题要么为真要么为假不存在第三种中间状态这是经典逻辑的排中律",
        "category": AXIOM, "depth": 3.0,
    },
    {
        "name": "矛盾律",
        "text": "一个命题和它的否定不能同时为真这是逻辑学的矛盾律",
        "category": AXIOM, "depth": 3.0,
    },
    {
        "name": "四则运算",
        "text": "加法 减法 乘法 除法 是基础的四则算术运算遵循交换律结合律分配律",
        "category": AXIOM, "depth": 2.5,
    },
    {
        "name": "等量代换",
        "text": "如果 A 等于 B 那么在任何式子中 A 都可以替换成 B 结果不变",
        "category": AXIOM, "depth": 2.5,
    },

    # ============ 二、物理与自然规律（depth 高）============
    {
        "name": "能量守恒定律",
        "text": "能量既不会凭空产生也不会凭空消失只能从一种形式转化为另一种形式总量不变",
        "category": AXIOM, "depth": 3.0,
        "relations": {"互斥": ["永动机谬误"]},
    },
    {
        "name": "永动机谬误",
        "text": "存在一种不需要任何能量输入就能永远运转并对外做功的永动机这违背能量守恒",
        "category": AXIOM, "depth": 0.4,
        "relations": {"互斥": ["能量守恒定律"]},
    },
    {
        "name": "光速上限",
        "text": "在真空中光速约每秒三十万公里是宇宙中信息和物质运动速度的上限无法被超越",
        "category": AXIOM, "depth": 2.8,
    },
    {
        "name": "万有引力",
        "text": "任何有质量的物体之间都存在相互吸引的引力地球的引力使物体下落",
        "category": COMMON, "depth": 2.0,
    },
    {
        "name": "热力学第二定律",
        "text": "孤立系统的总熵不会减少热量自发地从高温物体流向低温物体不会自发逆流",
        "category": AXIOM, "depth": 2.8,
    },

    # ============ 三、地理与天文常识（depth 中）============
    {
        "name": "地球形状",
        "text": "地球是一个两极略扁的近似球体它自转产生昼夜并围绕太阳公转产生四季",
        "category": COMMON, "depth": 2.2,
        "relations": {"互斥": ["地平说谬误"]},
    },
    {
        "name": "地平说谬误",
        "text": "地球是一个平面边缘是悬崖这是早已被科学和航天观测彻底否定的错误认知",
        "category": COMMON, "depth": 0.4,
        "relations": {"互斥": ["地球形状"]},
    },
    {
        "name": "太阳东升西落",
        "text": "由于地球自西向东自转太阳每天从东方升起在西方落下",
        "category": COMMON, "depth": 2.0,
        "relations": {"互斥": ["太阳西升谬误"]},
    },
    {
        "name": "太阳西升谬误",
        "text": "太阳从西边升起从东边落下这与地球自转方向相反在现实中不成立",
        "category": COMMON, "depth": 0.4,
        "relations": {"互斥": ["太阳东升西落"]},
    },

    # ============ 四、物质与生活常识（depth 中）============
    {
        "name": "水的相变",
        "text": "水在标准大气压下零摄氏度结冰一百摄氏度沸腾是日常的物态变化常识",
        "category": COMMON, "depth": 1.8,
    },
    {
        "name": "生物需要能量",
        "text": "人和动物需要进食获取能量植物通过光合作用把光能转化为化学能维持生命",
        "category": COMMON, "depth": 1.6,
    },
    {
        "name": "时间单向性",
        "text": "时间只能从过去流向未来不能倒流人无法回到过去改变已经发生的事",
        "category": COMMON, "depth": 2.0,
    },

    # ============ 五、领域知识：软件与编程（depth 中低）============
    {
        "name": "编程基础概念",
        "text": "代码 程序 函数 变量 类 对象 算法 数据结构 是软件开发的核心概念",
        "category": DOMAIN, "depth": 1.2,
    },
    {
        "name": "数据库概念",
        "text": "数据库用于存储和检索数据表 行 列 索引 查询 事务是关系数据库的基本要素",
        "category": DOMAIN, "depth": 1.2,
    },
    {
        "name": "人工智能概念",
        "text": "机器学习 神经网络 训练 推理 向量 嵌入 是人工智能领域的常见术语",
        "category": DOMAIN, "depth": 1.2,
    },
    {
        "name": "自由能原理",
        "text": "最小自由能原理认为智能系统通过不断最小化对外部世界的预测误差也就是惊奇度来维持自身存在",
        "category": DOMAIN, "depth": 1.5,
    },

    # ============ 六、日常交流（depth 低）============
    {
        "name": "日常问候",
        "text": "你好 早上好 晚上好 在吗 最近怎么样 谢谢 再见 是人类日常交流的礼貌问候用语",
        "category": COMMON, "depth": 0.9,
    },
    {
        "name": "情感表达",
        "text": "开心 难过 生气 惊讶 喜欢 讨厌 是人类常见的情绪和情感表达",
        "category": COMMON, "depth": 0.9,
    },
]


def iter_seed_concepts():
    """逐条产出种子概念，供灌库脚本使用。"""
    yield from SEED_CONCEPTS
