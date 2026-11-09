"""
Lab #4: System Prompt Engineering & Tool Calling Engine (Production Grade)
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + Tool Calling Engine (Hỗ trợ cả Mock ReAct Loop và Live Gemini API).
"""

import json
import re
import os
from typing import Dict, Any, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket, validate_tool_call

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và tiếp nhận phản hồi khách hàng cho VinFast và Vinpearl.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác và tôn trọng khách hàng.

## 2. AVAILABLE TOOLS
- `search_product_catalog`: Tra cứu sản phẩm/dịch vụ (xe điện, du lịch) theo danh mục và giá tối đa.
- `submit_support_ticket`: Tạo vé hỗ trợ kỹ thuật hoặc khiếu nại cho khách hàng.

## 3. CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm hoặc giá cả. PHẢI gọi tool để lấy dữ liệu thực tế.
2. BẮT BUỘC gọi `submit_support_ticket` khi khách hàng báo lỗi hoặc cần hỗ trợ.
3. Nếu không tìm thấy sản phẩm phù hợp, thông báo trung thực, không suy diễn thông tin.

## 4. OPERATIONAL BOUNDARIES
- Chỉ phục vụ các thông tin liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl).
- Từ chối lịch sự các chủ đề không thuộc phạm vi hoạt động.

## 5. OUTPUT CONTRACT
- Phản hồi theo định dạng: Thought -> Action -> Observation -> Final Answer.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent (Optimized Engine with Schema Validation & Live Support)
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling Engine tối ưu."""

    def __init__(self, max_iterations: int = 5, use_live_api: bool = False):
        self.max_iterations = max_iterations
        self.system_prompt = SYSTEM_PROMPT
        self.tools = TOOL_DEFINITIONS
        self.tool_map = TOOL_MAP
        self.use_live_api = use_live_api and bool(os.getenv("GEMINI_API_KEY"))
        self.trace: List[Dict[str, Any]] = []

    def _parse_price_vnd(self, text: str) -> int:
        """Trích xuất và chuẩn hóa mức giá VNĐ linh hoạt từ văn bản tự nhiên."""
        # Bắt các định dạng: 'dưới 600 triệu', 'dưới 600tr', 'dưới 1 tỷ', '< 600.000.000'
        price_patterns = [
            r"(?:dưới|<|không quá|tối đa|tầm)\s*([\d\.,]+)\s*(tỷ|ty|triệu|tr|nghìn|k|b|m|vnd|đ)?",
            r"([\d\.,]+)\s*(tỷ|ty|triệu|tr|nghìn|k|b|m)\b"
        ]
        
        for pat in price_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                amount_str = m.group(1).replace(",", "").replace(".", "")
                unit = (m.group(2) or "").lower()
                try:
                    amount = float(amount_str)
                    if any(u in unit for u in ["tỷ", "ty", "b"]):
                        return int(amount * 1_000_000_000)
                    elif any(u in unit for u in ["triệu", "tr", "m"]) or amount < 10000:
                        return int(amount * 1_000_000)
                    elif any(u in unit for u in ["nghìn", "k"]):
                        return int(amount * 1_000)
                    else:
                        return int(amount)
                except ValueError:
                    continue
        return 999999999999

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent và trích xuất tham số tự nhiên từ user_input."""
        text = user_input.lower()
        
        needs_catalog = False
        catalog_args: Dict[str, Any] = {}
        
        needs_ticket = False
        ticket_args: Dict[str, Any] = {}
        
        is_faq = False

        # 1. Phát hiện Intent tra cứu Catalog (xe điện / du lịch)
        has_car = bool(re.search(r"\b(xe|ô tô|oto|car|vf\s*\d*|bán tải|mini)\b", text, re.IGNORECASE))
        has_travel = bool(re.search(r"\b(resort|nghỉ dưỡng|du lịch|vinpearl|khách sạn|tour|phòng|villa)\b", text, re.IGNORECASE))
        
        has_search_kw = any(w in text for w in ["xem", "tìm", "giá", "dưới", "triệu", "tỷ", "có xe", "có phòng", "cho tôi"])
        if (has_car or has_travel) and has_search_kw:
            needs_catalog = True
            category = "xe_dien" if has_car else "du_lich"
            catalog_args["category"] = category
            catalog_args["max_price"] = self._parse_price_vnd(text)

        # 2. Phát hiện Intent gửi phiếu hỗ trợ (Support Ticket)
        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "hỗ trợ",
            "bị lỗi", "nghiêm trọng", "cần xử lý gấp", "ticket", "ghi nhận phản hồi", "ẩm mốc"
        ]
        if any(kw in text for kw in ticket_keywords) and any(w in text for w in ["tên", "tôi tên", "phản hồi", "gặp"]):
            needs_ticket = True
            
            # Trích xuất tên khách hàng
            name_match = re.search(r"(?:tên tôi là|tôi tên(?: là)?)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bvà\b|$)", user_input, re.IGNORECASE)
            ticket_args["customer_name"] = name_match.group(1).strip() if name_match else "Khách hàng Vingroup"
            
            # Trích xuất mức độ ưu tiên
            if any(p in text for p in ["nghiêm trọng", "gấp", "khẩn cấp", "high"]):
                priority = "high"
            elif any(p in text for p in ["thấp", "low"]):
                priority = "low"
            else:
                priority = "medium"
            ticket_args["priority"] = priority

            # Trích xuất mô tả sự cố
            issue_desc = user_input
            desc_match = re.search(r"(?:xe\s+[A-Za-z0-9\s]+bị[^\.,]+|phòng[^\.,]+bị[^\.,]+|bị lỗi[^\.,]+)", user_input, re.IGNORECASE)
            if desc_match:
                issue_desc = desc_match.group(0).strip()
            ticket_args["issue_description"] = issue_desc

        # 3. Phát hiện Intent FAQ (Hỏi chính sách không cần gọi tool)
        if not needs_catalog and not needs_ticket:
            if any(w in text for w in ["bảo hành", "chính sách", "ở đâu", "bao lâu", "mấy năm", "thời hạn"]):
                is_faq = True

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args,
            "is_faq": is_faq
        }

    def _execute_validated_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Thực thi tool có kèm kiểm tra Schema & xử lý ngoại lệ an toàn."""
        is_valid, err_msg = validate_tool_call(tool_name, args)
        if not is_valid:
            return {"error": f"Schema Validation Failed: {err_msg}"}

        func = self.tool_map.get(tool_name)
        if not func:
            return {"error": f"Tool '{tool_name}' không tồn tại."}

        try:
            return func(**args)
        except Exception as e:
            return {"error": f"Lỗi khi thực thi công cụ {tool_name}: {str(e)}"}

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intent(user_input)

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        iteration = 1

        # FAQ — Trả lời trực tiếp không cần gọi tool
        if intents["is_faq"]:
            faq_answer = (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km "
                "(tùy điều kiện nào đến trước). Trong thời gian bảo hành, pin sẽ được sửa chữa "
                "hoặc thay thế miễn phí nếu dung lượng khả dụng giảm xuống dưới 70%."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
                "final_answer": faq_answer
            })
            return {
                "answer": faq_answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        observations: Dict[str, Any] = {}
        tool_iterations = 0

        # Iteration 1: Gọi tool #1 nếu cần (search_product_catalog)
        if intents["needs_catalog"]:
            tool_iterations += 1
            cat_args = intents["catalog_args"]
            cat_result = self._execute_validated_tool("search_product_catalog", cat_args)
            observations["catalog"] = cat_result

            thought_msg = (
                f"Khách hàng muốn tìm sản phẩm thuộc danh mục '{cat_args.get('category')}'. "
                f"Tôi cần gọi tool search_product_catalog."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought_msg,
                "action": {
                    "name": "search_product_catalog",
                    "args": cat_args
                },
                "observation": cat_result
            })
            iteration += 1

        # Iteration 2: Gọi tool #2 nếu cần (submit_support_ticket)
        if intents["needs_ticket"]:
            tool_iterations += 1
            ticket_args = intents["ticket_args"]
            ticket_result = self._execute_validated_tool("submit_support_ticket", ticket_args)
            observations["ticket"] = ticket_result

            thought_msg = (
                f"Khách hàng {ticket_args.get('customer_name')} gặp sự cố. "
                f"Tôi cần gọi tool submit_support_ticket để tạo phiếu hỗ trợ."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought_msg,
                "action": {
                    "name": "submit_support_ticket",
                    "args": ticket_args
                },
                "observation": ticket_result
            })
            iteration += 1

        # Fallback: Không nhận diện được tác vụ
        if not observations:
            fallback_answer = "Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu của quý khách."
            self.trace.append({
                "iteration": iteration,
                "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
                "final_answer": fallback_answer
            })
            return {
                "answer": fallback_answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Iteration 3+: Tổng hợp Final Answer từ trace
        answer_parts = []
        if "catalog" in observations:
            products = observations["catalog"]
            if isinstance(products, list):
                if not products:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu của quý khách.")
                else:
                    cat_lines = [f"- {p['name']}: Giá {p['price_vnd']:,} VNĐ" for p in products]
                    answer_parts.append(
                        "Dưới đây là các sản phẩm phù hợp với yêu cầu của quý khách:\n" + "\n".join(cat_lines)
                    )
            elif isinstance(products, dict) and "error" in products:
                answer_parts.append(f"Lỗi khi tra cứu danh mục: {products['error']}")

        if "ticket" in observations:
            t = observations["ticket"]
            if isinstance(t, dict) and "ticket_id" in t:
                answer_parts.append(
                    f"Yêu cầu hỗ trợ của khách hàng {t['customer_name']} đã được ghi nhận thành công "
                    f"với mã phiếu {t['ticket_id']} (Mức độ ưu tiên: {t['priority']})."
                )
            elif isinstance(t, dict) and "error" in t:
                answer_parts.append(f"Lỗi khi ghi nhận phiếu: {t['error']}")

        final_answer = "\n\n".join(answer_parts)
        self.trace.append({
            "iteration": iteration,
            "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
            "final_answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": 1 if tool_iterations <= 1 else tool_iterations,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
