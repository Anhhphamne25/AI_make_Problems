from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict, TypedDict, Optional, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# Kiểu dữ liệu trả về mẫu
class ProgrammingProblem(BaseModel):
    name: str
    problemStatement: str
    inputFormat: str
    outputFormat: str
    constraints: str
    note: str = ""

    # solution: str = ""

    difficulty: str

    tags: List[str] = Field(default_factory=list)
    multipleFiles: Dict = Field(default_factory=dict)
    languages: List[str] = Field(default_factory=list)

    status: str = "draft"


class ProgrammingCritic(BaseModel):
    status: Literal["Approved", "Rejected"]
    feedback: str


# mô hình ai cho tạo đề và phản biện
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
).with_structured_output(ProgrammingProblem)

critic_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
).with_structured_output(ProgrammingCritic)

# prompt cho agent tạo đề và ai phản biện
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Bạn là GIẢNG VIÊN CNTT chuyên ra đề bài lập trình cho sinh viên đại học.

        NHIỆM VỤ:
        Sinh ra MỘT đề bài lập trình hoàn chỉnh, rõ ràng, đúng chuẩn học thuật.

        YÊU CẦU BẮT BUỘC:
        1. Tên bài ngắn gọn, phản ánh đúng nội dung bài toán
        2. Mô tả bài toán rõ ràng, không mơ hồ, không suy đoán
        3. Input format trình bày rõ từng dòng, từng biến
        4. Output format chính xác, không nhập nhằng
        5. Constraints đầy đủ và PHÙ HỢP với nội dung bài
        6. Ngôn ngữ học thuật, phù hợp môi trường đại học
        7. Đề bài phải TỰ ĐỦ – không cần giả định ngầm

        ĐỘ KHÓ (BẮT BUỘC TỰ ĐÁNH GIÁ):
        - easy   : vòng lặp, điều kiện, mảng cơ bản
        - medium : kết hợp nhiều cấu trúc, tư duy thuật toán
        - hard   : thuật toán nâng cao, tối ưu, dữ liệu lớn

        QUY TẮC GÁN ĐỘ KHÓ:
        - Chỉ gán easy nếu bài giải được bằng kỹ thuật cơ bản
        - Chỉ gán medium nếu cần tư duy thuật toán rõ ràng
        - Chỉ gán hard nếu có yêu cầu tối ưu hoặc dữ liệu lớn
        - KHÔNG được gán độ khó thấp hơn bản chất bài toán

        LƯU Ý QUAN TRỌNG:
        - KHÔNG tự kiểm tra, KHÔNG tự phê duyệt
        - KHÔNG viết lời giải
        - KHÔNG viết ví dụ nếu không cần thiết
        - Chỉ sinh nội dung đề bài
        """
    ),
    ("human", "{topic}")
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Bạn là GIẢNG VIÊN CNTT rất KHÓ TÍNH, chuyên phản biện đề bài lập trình.

        NHIỆM VỤ:
        Đánh giá đề bài một cách NGHIÊM KHẮC và KHÁCH QUAN.

        CHECKLIST BẮT BUỘC (thiếu 1 mục → KHÔNG ĐƯỢC Approved):
        1. Tên bài có rõ ràng, đúng nội dung bài toán
        2. Mô tả bài toán đầy đủ, không mơ hồ, không suy đoán
        3. Input format rõ ràng từng dòng, từng giá trị
        4. Output format chính xác, không gây hiểu nhầm
        5. Constraints đầy đủ và hợp lý với bài toán
        6. Độ khó (difficulty) PHÙ HỢP với nội dung và yêu cầu

        QUY TẮC ĐÁNH GIÁ:
        - Nếu THIẾU hoặc YẾU bất kỳ mục nào → status = "Rejected"
        - Chỉ khi TẤT CẢ đạt → status = "Approved"
        - Phải chỉ rõ CỤ THỂ mục nào cần chỉnh
        - KHÔNG tự suy diễn hoặc bổ sung giúp đề bài
        - Đánh giá ngắn gọn, thẳng thắn, đúng trọng tâm

        FORMAT OUTPUT (BẮT BUỘC – JSON):
        {
          "status": "Approved" hoặc "Rejected",
          "feedback": "Nếu Approved: nhận xét ngắn gọn lý do đạt. Nếu Rejected: liệt kê góp ý, mỗi ý một dòng."
        }

        Đề bài cần đánh giá:
        {problem_text}
        """
    ),
    ("human", "{problem_text}")
])


# agent
class GeneratorAgent:
    def run(self, topic: str) -> ProgrammingProblem:
        chain = GENERATOR_PROMPT | llm
        return chain.invoke({"topic": topic})


class CriticAgent:
    def run(self, problem: ProgrammingProblem) -> ProgrammingCritic:
        chain = CRITIC_PROMPT | critic_llm
        return chain.invoke({
            "problem_text": problem.model_dump_json(ensure_ascii=False)
        })


# kiểu dữ liệu dùng chung
class GraphState(TypedDict):
    topic: str
    problem: Optional[ProgrammingProblem]
    critic_result: Optional[ProgrammingCritic]
    iteration: int


# các hàm của graph
def generate_node(state: GraphState):
    generator = GeneratorAgent()
    iteration = state.get("iteration", 0) + 1
    if state.get("critic_result") is None:
        problem = generator.run(state["topic"])
    else:
        problem = generator.run(
            f"""
            Chủ đề gốc:
            {state["topic"]}

            Góp ý cần chỉnh:
            {state["critic_result"].feedback}

            Hãy chỉnh lại đề bài cho đầy đủ, rõ ràng và đúng chuẩn học thuật.
            """
        )
    print("\n--- Generated Problem ---\n")
    print(problem.model_dump_json(indent=2, ensure_ascii=False))
    print("\n-------------------------\n")
    return {"problem": problem,
            "iteration": iteration}


def critic_node(state: GraphState):
    critic = CriticAgent()
    result = critic.run(state["problem"])
    print("\n--- Critic Feedback ---\n")
    print("STATUS:", result.status)
    print("FEEDBACK:")
    print(result.feedback)
    print("\n-----------------------\n")
    return {"critic_result": result}


def should_continue(state: GraphState) -> str:
    if state["iteration"] >= 3:
        return "end"
    if state["critic_result"].status == "Approved":
        return "end"
    return "revise"


# hàm xử lý chính
def generate_problem(topic: str) -> ProgrammingProblem:
    graph = StateGraph(GraphState)

    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "revise": "generate",
        }
    )

    app = graph.compile()

    result = app.invoke({
        "topic": topic,
        "problem": None,
        "critic_result": None,
        "iteration": 0,
    })

    return result["problem"]


if __name__ == "__main__":
    topic = input("Nhập chủ đề bài tập lập trình: ")
    result = generate_problem(topic)

    print("\nCHATBOT OUTPUT:\n")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

# Hãy tạo một bài tập lập trình mức độ dễ về vòng lặp và số nguyên.
# Tạo bài tập lập trình về mảng một chiều, yêu cầu xử lý tần suất xuất hiện phần tử.
# Tạo một bài tập lập trình liên quan đến xử lý chuỗi, có kiểm tra điều kiện đặc biệt.
# Tạo bài tập lập trình về dãy con tăng dài nhất.
# Tạo bài tập lập trình yêu cầu sử dụng binary search để tối ưu kết quả.
# Tạo một bài tập lập trình bất kỳ.

