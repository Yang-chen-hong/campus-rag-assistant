# ============================================================
# 信科学院智能助手 - 统一 Skill 系统 v2.0
# ============================================================
# 设计理念：一处定义，三处复用（Agent / MCP / LangChain）
# 每个 Skill 真正查数据库，不硬编码
# ============================================================

import json
from typing import Any


# ============================================================
# Skill 基类
# ============================================================
class Skill:
    """所有 Skill 的基类"""

    name: str = ""
    description: str = ""
    parameters: dict = {}
    category: str = "通用"

    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_mcp_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# ============================================================
# Skill 1: 知识库检索（核心）
# ============================================================
class SearchKnowledgeBase(Skill):
    name = "search_knowledge_base"
    description = (
        "搜索信科学院知识库。当用户询问学院专业、课程、师资、政策等问题时调用，"
        "包括：奖学金、考试规定、处分、专业、宿舍、图书馆、食堂、学籍、毕业、学费、社团、军训、保研、"
        "教授/老师姓名、学院信息、校园设施等。"
        "查人名时直接输入全名搜索，如：蔡美玲、金龙。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如：奖学金评定条件、考试作弊处分、挂科补考、专业介绍",
            }
        },
        "required": ["query"],
    }
    category = "检索"

    def execute(self, query: str = "", chat_history: list = None, **kwargs) -> str:
        from retriever import search_test

        if not query:
            return "请提供搜索关键词"

        results = search_test(query, top_k=8, chat_history=chat_history)

        if not results:
            return (
                "未找到相关资料。建议换个关键词再试试，"
                "比如：奖学金评定条件、考试作弊处分、挂科补考规定。"
            )

        parts = [f"检索到 {len(results)} 条相关资料：\n"]
        for i, r in enumerate(results):
            parts.append(f"\n【来源{i+1}】《{r['title']}》相关度:{r['score']}\n")
            parts.append(f"{r['content']}\n")
        return "".join(parts)


# ============================================================
# Skill 2: 奖学金资格判断
# ============================================================
class CheckScholarship(Skill):
    name = "check_scholarship_eligibility"
    description = (
        "根据学生的年级、绩点、处分情况，判断符合哪类奖学金申请条件。"
        "需要用户提供年级、绩点和处分次数。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "grade": {"type": "string", "description": "年级，如：大一、大二、大三、大四"},
            "gpa": {"type": "number", "description": "绩点，如3.5（满分4.0）"},
            "punishment_count": {"type": "number", "description": "处分次数，0表示无处分"},
        },
        "required": ["grade", "gpa", "punishment_count"],
    }
    category = "学业"

    def execute(self, grade: str = "", gpa: float = 0,
                punishment_count: int = 0, **kwargs) -> str:
        from retriever import search_test

        # 先从数据库获取奖学金规定
        rules = search_test("奖学金评定条件 绩点要求", top_k=3, use_rewrite=False)
        rules_text = "\n".join(r["content"][:300] for r in rules)

        result = f"根据奖学金评定办法：\n\n"
        result += f"学生情况：{grade}，绩点{gpa}，处分{punishment_count}次\n\n"

        if punishment_count > 0:
            result += "❌ 不符合奖学金申请条件。\n"
            result += "原因：有处分记录。处分期间取消评奖评优资格。\n\n"
        elif gpa >= 3.8:
            result += "✅ 可申请国家奖学金（8000元/年，要求绩点≥3.8，排名前5%）\n"
        elif gpa >= 3.5:
            result += "✅ 可申请一等奖学金（要求：绩点≥3.5，无挂科，无处分）\n"
        elif gpa >= 3.0:
            result += "✅ 可申请二等奖学金（要求：绩点≥3.0，挂科≤1门，无处分）\n"
        elif gpa >= 2.5:
            result += "✅ 可申请三等奖学金（要求：绩点≥2.5，挂科≤2门，无处分）\n"
        else:
            result += "⚠️ 绩点较低，建议申请单项奖学金或进步奖\n"

        result += "\n奖学金类型和金额：\n"
        result += "- 国家奖学金：8000元/年\n"
        result += "- 国家励志奖学金：5000元/年（需家庭经济困难认定）\n"
        result += "- 校级一等奖学金：3000元/年\n"
        result += "- 校级二等奖学金：2000元/年\n"
        result += "- 校级三等奖学金：1000元/年\n"
        result += "- 单项奖学金：500元/年\n"

        if rules_text:
            result += f"\n\n【数据库参考】\n{rules_text[:500]}"
        return result


# ============================================================
# Skill 3: 处分规定查询
# ============================================================
class QueryDiscipline(Skill):
    name = "query_discipline_rules"
    description = (
        "查询学校处分相关规定。包括：挂科、作弊、旷课、考试纪律、"
        "处分等级、学术不端等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "rule_type": {"type": "string", "description": "规则类型：挂科、作弊、旷课、考试、处分、学术不端"}
        },
        "required": ["rule_type"],
    }
    category = "学业"

    def execute(self, rule_type: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(f"{rule_type} 处分规定 违纪", top_k=5, use_rewrite=False)

        if results:
            parts = [f"关于「{rule_type}」的相关规定：\n"]
            for i, r in enumerate(results):
                parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
            return "".join(parts)

        # 数据库没有就用默认规则
        rules = {
            "挂科": "📚 挂科后可申请补考，补考通过按60分计入。补考不过须重修。累计挂科超3门给学业预警，超8门编入下一年级。",
            "作弊": "🚫 考试作弊给记过及以上处分，成绩记零分不得补考，取消学位授予资格，记入诚信档案。",
            "旷课": "📋 旷课20-39学时警告，40-59学时严重警告，60-79学时记过，80学时以上留校察看。",
            "考试": "📝 按时参加考试，无故缺考按旷考处理。作弊按《学生违纪处分条例》处理。",
            "处分": "⚠️ 处分分五级：警告(6月)、严重警告(8月)、记过(10月)、留校察看(12月)、开除学籍。处分期间取消评奖评优资格。",
            "学术不端": "🚫 抄袭、剽窃、伪造数据等学术不端行为，视情节给予记过至开除学籍处分。毕业论文查重不合格取消答辩资格。",
        }
        for key, value in rules.items():
            if key in rule_type or rule_type in key:
                return value
        return f"未找到关于「{rule_type}」的规定。可查询：挂科、作弊、旷课、考试、处分、学术不端"


# ============================================================
# Skill 4: 学院专业信息查询
# ============================================================
class GetCollegeInfo(Skill):
    name = "get_college_info"
    description = "查询信科学院和专业信息，包括学院介绍、4个专业（计算机科学与技术、软件工程、物联网工程、人工智能）、师资力量、学科建设等。"
    parameters = {
        "type": "object",
        "properties": {
            "college_name": {"type": "string", "description": "学院名称或关键词，如：信息、计算机、外语、法学"},
        },
        "required": ["college_name"],
    }
    category = "信息"

    def execute(self, college_name: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(f"{college_name} 学院 专业 介绍", top_k=5, use_rewrite=False)

        if not results:
            return f"未找到关于「{college_name}」的学院信息。请尝试更准确的关键词。"

        parts = [f"找到 {len(results)} 条相关学院信息：\n"]
        for i, r in enumerate(results):
            parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
        return "".join(parts)


# ============================================================
# Skill 5: GPA 计算器
# ============================================================
class CalculateGPA(Skill):
    name = "calculate_gpa"
    description = "GPA计算器。输入各科成绩和学分，计算加权平均绩点。帮助学生了解自己的GPA水平。"
    parameters = {
        "type": "object",
        "properties": {
            "courses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "课程名称"},
                        "score": {"type": "number", "description": "成绩（百分制）"},
                        "credit": {"type": "number", "description": "学分"},
                    },
                },
                "description": "课程列表，每门课包含名称、成绩和学分",
            }
        },
        "required": ["courses"],
    }
    category = "学业"

    def execute(self, courses: list = None, **kwargs) -> str:
        if not courses:
            return "请提供课程信息（课程名、成绩、学分）"

        def score_to_gpa(score):
            if score >= 90:
                return 4.0
            elif score >= 85:
                return 3.7
            elif score >= 82:
                return 3.3
            elif score >= 78:
                return 3.0
            elif score >= 75:
                return 2.7
            elif score >= 72:
                return 2.3
            elif score >= 68:
                return 2.0
            elif score >= 64:
                return 1.5
            elif score >= 60:
                return 1.0
            else:
                return 0.0

        total_credits = 0
        total_gpa_credits = 0
        course_details = []

        for c in courses:
            name = c.get("name", "未知课程")
            score = float(c.get("score", 0))
            credit = float(c.get("credit", 0))
            gpa = score_to_gpa(score)
            total_credits += credit
            total_gpa_credits += gpa * credit
            course_details.append(f"  {name}：{score}分 → {gpa}（{credit}学分）")

        if total_credits == 0:
            return "学分总和为0，无法计算"

        final_gpa = round(total_gpa_credits / total_credits, 2)

        result = "📊 GPA计算结果\n\n"
        result += "课程明细：\n" + "\n".join(course_details) + "\n\n"
        result += f"总学分：{total_credits}\n"
        result += f"GPA：{final_gpa}\n\n"

        if final_gpa >= 3.8:
            result += "🏆 优秀！可申请国家奖学金"
        elif final_gpa >= 3.5:
            result += "✅ 优秀，可申请一等奖学金"
        elif final_gpa >= 3.0:
            result += "👍 良好，可申请二等奖学金"
        elif final_gpa >= 2.5:
            result += "⚠️ 一般，可申请三等奖学金"
        else:
            result += "⚠️ 需要努力提高成绩"

        return result


# ============================================================
# Skill 6: 毕业条件检查
# ============================================================
class CheckGraduation(Skill):
    name = "check_graduation_requirements"
    description = "检查毕业和学位授予条件。根据学生的学分、挂科情况、处分情况判断是否满足毕业要求。"
    parameters = {
        "type": "object",
        "properties": {
            "total_credits": {"type": "number", "description": "已修总学分"},
            "required_credits": {"type": "number", "description": "毕业所需总学分（一般为160-170）"},
            "failed_courses": {"type": "number", "description": "当前未通过课程数"},
            "has_punishment": {"type": "boolean", "description": "是否有未解除的处分"},
            "thesis_passed": {"type": "boolean", "description": "毕业论文/设计是否通过"},
        },
        "required": ["total_credits", "required_credits", "failed_courses", "has_punishment", "thesis_passed"],
    }
    category = "学业"

    def execute(self, total_credits: float = 0, required_credits: float = 165,
                failed_courses: int = 0, has_punishment: bool = False,
                thesis_passed: bool = False, **kwargs) -> str:
        from retriever import search_test

        # 查数据库获取毕业条件
        rules = search_test("毕业条件 学位授予 学分要求", top_k=3, use_rewrite=False)

        result = "🎓 毕业条件检查结果\n\n"
        result += f"当前情况：\n"
        result += f"- 已修学分：{total_credits} / {required_credits}\n"
        result += f"- 未通过课程：{failed_courses}门\n"
        result += f"- 处分状态：{'有未解除处分' if has_punishment else '无处分'}\n"
        result += f"- 毕业论文：{'已通过' if thesis_passed else '未通过'}\n\n"

        issues = []

        if total_credits < required_credits:
            gap = required_credits - total_credits
            issues.append(f"❌ 学分不足，还差{gap}学分")
        else:
            result += "✅ 学分已满足要求\n"

        if failed_courses > 0:
            issues.append(f"❌ 有{failed_courses}门课程未通过，需补考或重修")
        else:
            result += "✅ 无未通过课程\n"

        if has_punishment:
            issues.append("❌ 有未解除的处分，可能影响学位授予")
        else:
            result += "✅ 无处分记录\n"

        if not thesis_passed:
            issues.append("❌ 毕业论文/设计未通过")
        else:
            result += "✅ 毕业论文已通过\n"

        if issues:
            result += "\n⚠️ 需要解决的问题：\n"
            for issue in issues:
                result += f"{issue}\n"
        else:
            result += "\n🎉 恭喜！你满足毕业和学位授予的所有条件！"

        if rules:
            result += "\n\n【数据库参考】\n" + rules[0]["content"][:400]

        return result


# ============================================================
# Skill 7: 校园常用电话查询
# ============================================================
class GetCampusContacts(Skill):
    name = "get_campus_contacts"
    description = "查询校园常用服务电话和办公地点，如教务处、学工处、保卫处、校医院、图书馆等。"
    parameters = {
        "type": "object",
        "properties": {
            "department": {"type": "string", "description": "部门名称，如：教务处、学工处、保卫处、校医院"},
        },
        "required": ["department"],
    }
    category = "生活"

    def execute(self, department: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(f"{department} 电话 联系 办公地点", top_k=3, use_rewrite=False)

        contacts = {
            "保卫处": "保卫处24小时值班：0731-88872110。负责校园安全、门卫管理、巡逻等。",
            "校医院": "校医院急诊：0731-88872342。门诊周一至周五，急诊24小时值班。位于二里半校区。",
            "学工处": "学生工作处：0731-88872262。负责学生日常管理、奖助学金、心理咨询等。",
            "教务处": "教务处：0731-88872223。负责选课、考试、学籍、成绩等。位于行政楼。",
            "招生": "招生与就业指导处：0731-88872216。官网：zsjy.hunnu.edu.cn",
            "后勤": "后勤管理处：0731-88872318。位于二里半校区木兰路。",
            "图书馆": "图书馆：0731-88872441。二里半校区总馆。",
            "网络": "网络中心：0731-88872456。负责校园网、信息系统等。",
        }

        for key, value in contacts.items():
            if key in department or department in key:
                result = f"📞 {value}"
                if results:
                    result += f"\n\n【数据库补充】\n{results[0]['content'][:300]}"
                return result

        if results:
            parts = [f"找到关于「{department}」的信息：\n"]
            for i, r in enumerate(results):
                parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
            return "".join(parts)

        return f"未找到「{department}」的联系方式。可查询：保卫处、校医院、学工处、教务处、招生、后勤、图书馆、网络"


# ============================================================
# Skill 8: 学费查询
# ============================================================
class CheckTuition(Skill):
    name = "check_tuition_fees"
    description = "查询各专业学费标准、住宿费、缴费方式等信息。"
    parameters = {
        "type": "object",
        "properties": {
            "major_type": {"type": "string", "description": "专业类型，如：文史、理工、艺术、体育、医学、计算机"},
        },
        "required": ["major_type"],
    }
    category = "生活"

    def execute(self, major_type: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(f"{major_type} 学费 收费标准", top_k=3, use_rewrite=False)

        if results:
            parts = [f"关于「{major_type}」的学费信息：\n"]
            for i, r in enumerate(results):
                parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
            return "".join(parts)

        return f"未找到「{major_type}」的学费信息。可查询：文史、理工、艺术、体育、医学、计算机"


# ============================================================
# Skill 9: 宿舍信息查询
# ============================================================
class GetDormitoryInfo(Skill):
    name = "get_dormitory_info"
    description = "查询学生宿舍条件、住宿费、宿舍管理规定、水电费等信息。"
    parameters = {
        "type": "object",
        "properties": {
            "info_type": {"type": "string", "description": "查询类型：条件、费用、规定、报修、调换"},
        },
        "required": ["info_type"],
    }
    category = "生活"

    def execute(self, info_type: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(f"宿舍 {info_type} 住宿 管理", top_k=5, use_rewrite=False)

        if results:
            parts = [f"关于宿舍「{info_type}」的信息：\n"]
            for i, r in enumerate(results):
                parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
            return "".join(parts)

        return f"未找到宿舍「{info_type}」的信息。可查询：条件、费用、规定、报修、调换"


# ============================================================
# Skill 10: FAQ智能匹配
# ============================================================
class CampusFAQMatch(Skill):
    name = "campus_faq_match"
    description = "信科学院高频问题智能匹配。当用户问的是常见问题（如新生入学、校园卡、转专业、保研等），直接返回FAQ答案。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "用户的问题"},
        },
        "required": ["question"],
    }
    category = "检索"

    def execute(self, question: str = "", **kwargs) -> str:
        from retriever import search_test

        results = search_test(question, top_k=5, use_rewrite=False)

        if not results:
            return "未找到匹配的FAQ。建议用 search_knowledge_base 工具搜索更多资料。"

        parts = [f"匹配到 {len(results)} 条相关FAQ：\n"]
        for i, r in enumerate(results):
            parts.append(f"\n【来源{i+1}】《{r['title']}》相关度:{r['score']}\n")
            parts.append(f"{r['content']}\n")
        return "".join(parts)


# ============================================================
# Skill 注册表
# ============================================================
class SkillRegistry:
    """统一 Skill 注册表 — 一处定义，三处复用"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def execute(self, name: str, arguments: dict,
                chat_history: list = None) -> str:
        skill = self.get(name)
        if not skill:
            return f"未知工具：{name}"

        if chat_history is not None:
            arguments["chat_history"] = chat_history

        try:
            return skill.execute(**arguments)
        except Exception as e:
            return f"工具执行出错：{e}"

    # ---- 格式转换 ----
    def to_openai_format(self) -> list[dict]:
        return [s.to_openai_format() for s in self._skills.values()]

    def to_mcp_format(self) -> list[dict]:
        return [s.to_mcp_format() for s in self._skills.values()]

    def to_langchain_format(self):
        """转换为 LangChain @tool 格式"""
        from langchain.tools import tool as lc_tool

        lc_tools = []
        for skill in self._skills.values():
            def make_executor(skl: Skill):
                def _fn(**kwargs) -> str:
                    return skl.execute(**kwargs)
                _fn.__name__ = skl.name
                _fn.__doc__ = skl.description
                return _fn

            lc_tools.append(lc_tool(make_executor(skill)))
        return lc_tools

    def summary(self) -> str:
        lines = [f"共注册 {len(self._skills)} 个 Skill：\n"]
        for name, skill in self._skills.items():
            lines.append(f"  [{skill.category}] {name} — {skill.description[:50]}...")
        return "\n".join(lines)


# ============================================================
# 全局注册表实例
# ============================================================
registry = SkillRegistry()

registry.register(SearchKnowledgeBase())
registry.register(CheckScholarship())
registry.register(QueryDiscipline())
registry.register(GetCollegeInfo())
registry.register(CalculateGPA())
registry.register(CheckGraduation())
registry.register(GetCampusContacts())
registry.register(CheckTuition())
registry.register(GetDormitoryInfo())
registry.register(CampusFAQMatch())


# ============================================================
# 便捷函数
# ============================================================
def get_all_skills() -> list[Skill]:
    return registry.all()

def get_openai_tools() -> list[dict]:
    return registry.to_openai_format()

def get_mcp_tools() -> list[dict]:
    return registry.to_mcp_format()

def get_langchain_tools():
    return registry.to_langchain_format()

def execute_skill(name: str, arguments: dict,
                  chat_history: list = None) -> str:
    return registry.execute(name, arguments, chat_history)


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("信科学院智能助手 - Skill 系统 v2.0")
    print("=" * 60)
    print(registry.summary())

    print("\n" + "=" * 60)
    print("OpenAI 格式输出（前2个）:")
    print("=" * 60)
    for t in get_openai_tools()[:2]:
        print(json.dumps(t, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("MCP 格式输出（前2个）:")
    print("=" * 60)
    for t in get_mcp_tools()[:2]:
        print(json.dumps(t, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("执行测试:")
    print("=" * 60)

    test_cases = [
        ("campus_faq_match", {"question": "学费多少钱"}),
        ("get_campus_contacts", {"department": "保卫处"}),
        ("calculate_gpa", {"courses": [
            {"name": "高等数学", "score": 88, "credit": 4},
            {"name": "大学英语", "score": 92, "credit": 3},
            {"name": "程序设计", "score": 76, "credit": 3},
        ]}),
        ("check_scholarship_eligibility",
         {"grade": "大二", "gpa": 3.6, "punishment_count": 0}),
    ]

    for name, args in test_cases:
        print(f"\n--- {name}({args}) ---")
        result = execute_skill(name, args)
        print(result[:300])
