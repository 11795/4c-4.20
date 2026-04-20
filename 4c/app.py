import os
from typing import Annotated, Sequence, TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()
qwen_key = os.getenv("DASHSCOPE_API_KEY")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid # 用于生成唯一的会话ID
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware # 解决前端跨域问题
from pydantic import BaseModel # 定义数据格式
import uvicorn # 运行服务器

from pathlib import Path

# 确保路径在云端是绝对路径
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = str(BASE_DIR / "local_faiss_index")

# ... (init_rag_knowledge_base 函数内加载路径确保使用 VECTOR_DB_DIR) ...

def init_rag_knowledge_base():
    """初始化或加载 RAG 向量数据库"""
    print("正在检查知识库状态...")
    embeddings = DashScopeEmbeddings(dashscope_api_key=qwen_key)

    # 判断本地是否已经存在建好的向量库
    if os.path.exists(VECTOR_DB_DIR):
        print("-> 发现本地向量库缓存，直接加载 (极速模式) ⚡")
        # allow_dangerous_deserialization=True 是新版 LangChain 要求的安全参数，用于加载本地 FAISS
        vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("-> 未发现缓存，正在读取心理学文档并构建向量数据库... ⏳")
        # 1. 加载单独的 TXT 文档 (务必指定 encoding="utf-8" 防止中文乱码)
        loader = TextLoader("psychology_knowledge.txt", encoding="utf-8")
        docs = loader.load()

        # 2. 文本分块 (Chunking)
        # chunk_size: 每次切多大；chunk_overlap: 上下块重叠多少字以保持上下文连贯
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # 3. 调用 Qwen Embedding 模型进行向量化，并存入 FAISS
        vector_db = FAISS.from_documents(split_docs, embeddings)

        # 4. 保存到本地文件夹，下次运行就不用重新花钱/花时间向量化了！
        vector_db.save_local(VECTOR_DB_DIR)
        print("-> 向量数据库构建并持久化完成！✅")

    # 返回检索器 (Retriever)，每次返回最相关的 1 个知识块 (k=1)
    return vector_db.as_retriever(search_kwargs={"k": 1})


# 在程序启动时初始化检索器
retriever = init_rag_knowledge_base()

# ==========================================
# 1. 定义系统状态 (State)
# ==========================================
# Annotated[..., add_messages] 会自动将新消息追加到历史记录中，形成原生记忆机制
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    crisis_flag: str  # 记录当前轮次是否触发危机

# 定义请求体：用户发来的消息
class ChatInput(BaseModel):
    message: str
    thread_id: str = None  # 如果不传，则视为新会话

# 定义返回体：返回给前端的消息
class ChatOutput(BaseModel):
    reply: str
    thread_id: str

# ==========================================
# 2. 定义专业干预工具 (Tools)
# ==========================================
@tool
def grounding_exercise_tool() -> str:
    """当用户感到极度焦虑、恐慌发作、情绪失控或感觉虚幻时调用此工具。"""
    return """【干预提示被触发：着陆技术 (Grounding)】
    请温柔地引导用户进行 5-4-3-2-1 练习：
    告诉用户：“我们一起来做一个小练习，把注意力带回当下。请环顾四周，告诉我：
    5件你能看到的东西，4件你能摸到的东西，3件你能听到的声音，2件你能闻到的气味，1件你能尝到的味道。”"""


@tool
def cbt_thought_record_tool() -> str:
    """当用户陷入极度负面、绝对化思维（如“我什么都做不好”、“所有人讨厌我”）时调用。"""
    return """【干预提示被触发：CBT认知解离】
    请温和地引导用户质疑这个想法，提问方向（每次选一个）：
    1. 有什么客观证据支持你这个想法吗？有什么证据反对吗？
    2. 如果你的好朋友遇到完全一样的情况，你会对TA说什么？
    注意：不要直接说教，用苏格拉底式提问启发用户。"""

@tool
def search_psychology_knowledge(query: str) -> str:
    """
    【核心工具：综合知识检索】
    无论用户是在探讨情绪困扰（心理学），还是咨询职业规划、健康养生、人际交往、学习方法、生活常识等其他领域的问题，
    只要你需要给出具体的建议、方法或科普解答，**必须主动且优先调用此工具**去查找依据！
    绝不允许完全依赖你自己的内部数据去讲道理。
    输入参数 query：提取用户话语中的核心关键词（如"缓解失眠"、"职场面试技巧"、"时间管理法则"、"颈椎病预防"）。
    """
    print(f"\n[系统日志]: 正在检索心理学知识库，关键词：{query}...")
    docs = retriever.invoke(query)
    if docs:
        result = "检索到的专业心理学知识：\n" + "\n".join([doc.page_content for doc in docs])
        result += "\n【系统指令】：请将上述专业知识用温暖、共情的口吻传达给用户，避免生硬说教。"
        return result
    return "没有检索到相关的心理学技巧，请使用一般性的共情话语进行安抚。"

tools = [grounding_exercise_tool, cbt_thought_record_tool, search_psychology_knowledge]

# ==========================================
# 3. 初始化 LLM 和 Prompt
# ==========================================
# 危机检测使用低配模型以保证极速响应；咨询使用高配模型保证共情能力。

crisis_llm = ChatTongyi(model='qwen-turbo', api_key=qwen_key, temperature=0.1)
counseling_llm = ChatTongyi(model='qwen3-max', api_key=qwen_key, temperature=0.7)

counseling_llm_with_tools = counseling_llm.bind_tools(tools)

SYSTEM_PROMPT = """你是一个全能的AI生活教练与心理支持助手，名字叫“Lumos”。
你不仅能提供温暖的情感支持，还能解答生活、职场、健康、学习等多方面的实际问题。

【核心原则】
1. 共情与接纳：在回答问题前，先用一两句话回应用户的情绪或处境，展现温暖和理解。
2. 跨领域知识支持：【最高优先级】遇到用户寻求具体方法、常识、建议（无论是心理学、职场、生活技能等）时，**你必须主动思考并调用 `search_psychology_knowledge` 工具**。
3. 自然表达：将检索到的知识用口语化、像朋友聊天一样的语气表达出来，绝对不要像在背说明书。
4. 语言要求：每次回复控制在100字到150字左右，简明扼要。每次只给出一个核心建议或提一个小问题。

【严禁行为】
- 严禁对用户进行精神疾病诊断或开药。
- 严禁在没有调用知识库工具的情况下，擅自长篇大论地进行生硬的说教。
"""


# ==========================================
# 4. 定义节点 (Nodes)
# ==========================================
def evaluate_crisis(state: AgentState):
    """节点1：危机守卫，每句输入必经的第一道关卡"""
    last_message = state["messages"][-1].content
    prompt = f"""分析以下用户输入，是否包含自杀、自残、严重抑郁、绝望或伤害他人的明确意图。
    只回答 'YES' 或 'NO'。
    用户输入: {last_message}"""

    response = crisis_llm.invoke(prompt)
    flag = "YES" if "YES" in response.content.upper() else "NO"
    return {"crisis_flag": flag}


def chat_node(state: AgentState):
    """节点2：主咨询大脑"""
    messages = list(state["messages"])
    # 确保 SystemPrompt 始终在最前面
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    response = counseling_llm_with_tools.invoke(prompt_messages)
    return {"messages": [response]}


def sos_node(state: AgentState):
    """节点3：危机熔断拦截器"""
    sos_msg = """【系统触发安全协议】
我听到你现在非常痛苦，我非常担心你的安全。作为AI我无法握住你的手，但请立刻联系真实世界中能帮助你的人：
- 中国24小时心理危机干预热线：400-161-9995
- 报警及急救电话：110 / 120
你并不孤单，你的生命非常重要，请给专业人士一个帮助你的机会。"""
    return {"messages": [AIMessage(content=sos_msg)]}


# ==========================================
# 5. 定义路由逻辑 (Conditional Edges)
# ==========================================
def route_after_evaluation(state: AgentState) -> str:
    """根据危机标志决定是进入咨询还是触发警报"""
    if state["crisis_flag"] == "YES":
        return "sos_node"
    return "chat_node"


def route_after_chat(state: AgentState) -> str:
    """检查模型是否决定调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools_node"
    return END


# ==========================================
# 6. 构建 LangGraph 状态机
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("evaluate_crisis", evaluate_crisis)
workflow.add_node("chat_node", chat_node)
workflow.add_node("sos_node", sos_node)
workflow.add_node("tools_node", ToolNode(tools))  # 内置工具节点处理调用结果

# 绘制边（连接流程）
workflow.add_edge(START, "evaluate_crisis")
workflow.add_conditional_edges("evaluate_crisis", route_after_evaluation)
workflow.add_conditional_edges("chat_node", route_after_chat)
workflow.add_edge("tools_node", "chat_node")  # 工具执行完后回到 chat_node 总结
workflow.add_edge("sos_node", END)  # SOS 触发后直接结束当前轮次

# 编译 Graph，并加入持久化记忆 (MemorySaver)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

fastapi_app = FastAPI(title="Lumos心理咨询 API")

# 允许跨域（如果你要写前端网页调用这个接口，这一步必不可少）
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议改为具体的域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 7. 终端交互界面
# ==========================================
@fastapi_app.post("/chat", response_model=ChatOutput)
async def chat(input_data: ChatInput):
        # 1. 确定会话 ID
        thread_id = input_data.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        user_message = input_data.message
        final_reply = ""

        try:
            # 2. 调用 LangGraph 运行流
            # 我们迭代所有的事件，但最终只抓取 chat_node 或 sos_node 产生的最后一句 AIMessage
            events = app.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="values"  # 获取完整的 state 变化
            )

            for event in events:
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    # 只有当最后一条消息是 AI 发出的，且有内容时，我们才记录它
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        final_reply = last_msg.content

            # 3. 返回给前端
            return ChatOutput(reply=final_reply, thread_id=thread_id)

        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail="服务器内部错误")

if __name__ == "__main__":
    # 云函数环境通常从环境变量获取端口，默认 9000
    port = int(os.environ.get("FC_SERVER_PORT", 9000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)