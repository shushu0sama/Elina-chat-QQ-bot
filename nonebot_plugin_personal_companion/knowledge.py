import yaml
import random
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).parent / "prompts" / "philosophy_knowledge.yaml"
MANIFESTATION_KNOWLEDGE_PATH = Path(__file__).parent / "prompts" / "manifestation_knowledge.yaml"


class KnowledgeBase:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else KNOWLEDGE_PATH
        self.data = self._load()
        self.concepts: list[dict] = self.data["concepts"]
        self.situations: list[dict] = self.data["situations"]

    def _load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def retrieve(self, user_msg: str, limit: int = 2) -> list[dict]:
        """Retrieve the most relevant concepts for a user message."""
        if not user_msg.strip():
            return []

        scored = self._score_concepts(user_msg)

        situation_matches = self._match_situations(user_msg)
        for sit in situation_matches:
            primary = sit["primary"]
            for name in scored:
                if primary in name:
                    scored[name] += 2
            for sec in sit.get("secondary", []):
                for name in scored:
                    if sec in name:
                        scored[name] += 1

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top = [name for name, score in ranked if score > 0][:limit]

        results: list[dict] = []
        for name in top:
            concept = self._get_concept_by_name(name)
            if concept:
                wisdom_pool = concept.get("wisdom", [])
                essence = concept.get("essence", "")
                n = min(2, len(wisdom_pool))
                selected = random.sample(wisdom_pool, n) if n > 0 else []
                results.append({
                    "name": concept["name"],
                    "essence": essence,
                    "wisdom": selected,
                    "avoid": concept.get("avoid", []),
                })

        return results

    def _score_concepts(self, user_msg: str) -> dict[str, int]:
        scores: dict[str, int] = {}
        msg_lower = user_msg.lower()
        for concept in self.concepts:
            name = concept["name"]
            score = 0
            for kw in concept.get("keywords", []):
                if kw in msg_lower or kw in user_msg:
                    score += 1
            if score > 0:
                scores[name] = score
        return scores

    def _match_situations(self, user_msg: str) -> list[dict]:
        matches: list[dict] = []
        for sit in self.situations:
            trigger_words = sit.get("triggers", [])
            if not trigger_words:
                trigger_words = {
                    "用户诉说负面情绪/焦虑/压力": ["焦虑", "压力", "难受", "害怕", "担心", "恐惧", "愤怒", "生气", "难过", "痛苦", "崩溃", "烦躁", "不开心", "伤心", "委屈", "压抑"],
                    "用户分享好事/好消息/开心的事": ["开心", "好消息", "幸运", "顺利", "成功", "拿到", "过了", "中了", "收到礼物", "好开心", "太棒了", "高兴", "庆祝"],
                    "用户反复诉说同一个困扰": ["又是", "还是", "又来了", "老是", "总是", "每次都这样", "又这样", "再一次"],
                    "用户谈到金钱问题/经济压力": ["钱", "工资", "没钱", "穷", "涨价", "消费", "收入", "还款", "房贷", "车贷", "账单", "不够用", "经济"],
                    "用户说'我不行/做不到'这类话": ["我不行", "做不到", "买不起", "我不够", "我没有", "我做不到", "我不配", "我没资格", "太难了"],
                    "用户感到迷茫/找不到方向": ["迷茫", "不知道该怎么办", "没有方向", "找不到", "看不清楚", "不知道该", "不知道要什么"],
                    "用户经历'巧合'/说不上来的直觉": ["巧合", "巧了", "太巧了", "说不上来", "直觉", "感觉对了", "莫名", "奇妙的"],
                    "用户在看电影/读书/听音乐有所感触": ["电影", "看书", "读书", "听歌", "音乐", "这本书", "那本书", "这个电影", "那部电影"],
                    "日常小牢骚/轻微不满": ["烦", "好烦", "好累", "无语", "服了", "没意思", "无聊", "不顺", "小事", "琐事", "烦人"],
                }.get(sit["situation"], [])
            if any(tw in user_msg for tw in trigger_words):
                matches.append(sit)
        return matches

    def _get_concept_by_name(self, name: str) -> dict | None:
        for c in self.concepts:
            if c["name"] == name:
                return c
        return None

    def get_situation_guidance(self, user_msg: str) -> str | None:
        situations = self._match_situations(user_msg)
        if situations:
            return situations[0].get("guidance")
        return None


def build_knowledge_prompt(user_msg: str, kb: KnowledgeBase) -> str:
    """Build a compact system prompt block with retrieved philosophy knowledge."""
    concepts = kb.retrieve(user_msg, limit=2)
    if not concepts:
        return ""

    lines = ["[与当前话题相关的核心视角——请自然地融入对话，不要照搬原话：]"]
    for c in concepts:
        lines.append(f"- {c['essence']}")
        for w in c.get("wisdom", []):
            lines.append(f"  → {w}")

    guidance = kb.get_situation_guidance(user_msg)
    if guidance:
        lines.append(f"（提示：{guidance}）")

    return "\n".join(lines)


def build_knowledge_prompt_personalized(user_msg: str, kb: KnowledgeBase,
                                         user_id: int, memory_store) -> str:
    """Build a knowledge prompt enriched with the user's own experiences."""
    concepts = kb.retrieve(user_msg, limit=2)
    if not concepts:
        return ""

    lines = ["[与当前话题相关的核心视角——请自然地融入对话，不要照搬原话：]"]
    for c in concepts:
        lines.append(f"- {c['essence']}")
        for w in c.get("wisdom", []):
            lines.append(f"  → {w}")

        concept_data = kb._get_concept_by_name(c["name"])
        if concept_data and memory_store:
            concept_kws = concept_data.get("keywords", [])[:5]
            personal = memory_store.retrieve_memories(concept_kws, user_id, limit=2)
            if personal:
                lines.append("  【关于这个视角，你记得对方经历过这些——如果自然可以轻轻提起：】")
                for mem in personal:
                    lines.append(f"  → 对方经历过：{mem}")
                lines.append("  如果话题自然相关，可以用'你记不记得你之前...'来关联，但不要强行扯到哲学概念。")

    guidance = kb.get_situation_guidance(user_msg)
    if guidance:
        lines.append(f"（提示：{guidance}）")

    return "\n".join(lines)


def is_manifestation_intent(user_msg: str) -> bool:
    markers = [
        "显化", "宇宙订单", "下订单", "愿望", "吸引力法则", "限制性信念", "证据",
        "未来自我", "未来的我", "感恩", "频率", "能量频道", "显化肌肉", "小魔女",
        "为什么还没发生", "我不配", "不可能", "不会轮到我", "想要", "我想拥有",
    ]
    return any(marker in user_msg for marker in markers)


def build_manifestation_knowledge_prompt(user_msg: str, kb: KnowledgeBase,
                                         user_id: int | None = None, memory_store=None) -> str:
    concepts = kb.retrieve(user_msg, limit=2)
    if not concepts:
        return ""

    lines = [
        "[显化知识视角——只在当前话题相关时自然融入，不要照搬术语：]",
        "安全边界：不承诺结果一定发生；不把未显化归咎于用户频率不够；不鼓励反复确认、追逐或控制他人；显化要落到愿望澄清、信念调整、状态稳定、证据收集和对齐行动。",
    ]
    for c in concepts:
        lines.append(f"- {c['essence']}")
        for w in c.get("wisdom", []):
            lines.append(f"  → {w}")

        concept_data = kb._get_concept_by_name(c["name"])
        if concept_data and memory_store and user_id is not None:
            concept_kws = concept_data.get("keywords", [])[:5]
            personal = memory_store.retrieve_memories(concept_kws, user_id, limit=2)
            if personal:
                lines.append("  【可自然关联的个人显化背景：】")
                for mem in personal:
                    lines.append(f"  → {mem}")

    if memory_store and user_id is not None:
        wishes = memory_store.get_manifestation_wishes(user_id, statuses=["active"], limit=3)
        evidence = memory_store.get_manifestation_evidence(user_id, limit=3)
        if wishes:
            lines.append("【当前 active 愿望，只有自然相关时使用，不要催结果：】")
            for wish in wishes:
                lines.append(f"- #{wish['id']} {wish['title']}")
        if evidence:
            lines.append("【最近显化证据链，可用于提醒用户已有进展：】")
            for item in evidence:
                lines.append(f"- {item['content']}")

    guidance = kb.get_situation_guidance(user_msg)
    if guidance:
        lines.append(f"（显化回应提示：{guidance}）")

    return "\n".join(lines)
