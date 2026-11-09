import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog (Optimized & Robust)
# ---------------------------------------------------------------------------

def search_product_catalog(
    category: str,
    max_price: int = 999999999999,
    keyword: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục, giá tối đa và từ khóa tùy chọn.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich'). Hỗ trợ chuẩn hóa tiếng Việt.
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
        keyword: Từ khóa tìm kiếm thêm trong tên hoặc mô tả (tùy chọn).
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện lọc.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return []

    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    # Chuẩn hóa danh mục (hỗ trợ cả dạng text tự nhiên có dấu)
    cat_norm = category.strip().lower()
    if cat_norm in ["xe", "xe điện", "ô tô", "oto", "car", "xe_dien"]:
        target_cat = "xe_dien"
    elif cat_norm in ["du lịch", "resort", "nghỉ dưỡng", "vinpearl", "khách sạn", "du_lich"]:
        target_cat = "du_lich"
    else:
        target_cat = cat_norm

    kw_norm = keyword.strip().lower() if keyword else None

    results = []
    for p in products:
        p_cat = p.get("category", "").lower()
        p_price = p.get("price_vnd", 0)

        if p_cat == target_cat and p_price <= max_price:
            if kw_norm:
                p_text = f"{p.get('name', '')} {p.get('description', '')}".lower()
                if kw_norm in p_text:
                    results.append(p)
            else:
                results.append(p)

    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket (Optimized & Thread-safe simulation)
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ/khiếu nại của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    existing_tickets = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                existing_tickets = json.load(f)
        except Exception:
            existing_tickets = []

    # Chuẩn hóa mức độ ưu tiên
    p_norm = priority.strip().lower()
    if p_norm in ["khẩn cấp", "gấp", "nghiêm trọng", "high"]:
        norm_priority = "high"
    elif p_norm in ["thấp", "low"]:
        norm_priority = "low"
    else:
        norm_priority = "medium"

    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "issue_description": issue_description.strip(),
        "priority": norm_priority,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    os.makedirs(os.path.dirname(tickets_file), exist_ok=True)
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "issue_description": issue_description.strip(),
        "priority": norm_priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas chuẩn OpenAPI 3.0 / Gemini Tool Use
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu danh mục sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch Vinpearl) theo ngân sách tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Danh mục sản phẩm cần tra cứu: 'xe_dien' hoặc 'du_lich'.",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": "Mức giá tối đa khách hàng có thể chi trả (tính theo VNĐ).",
                    "default": 999999999999
                },
                "keyword": {
                    "type": "string",
                    "description": "Từ khóa tìm kiếm bổ sung (ví dụ: 'VF 3', 'Nha Trang')."
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": "Tiếp nhận khiếu nại, phản ánh sự cố kỹ thuật của khách hàng và lưu trữ vào hệ thống ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ và tên đầy đủ của khách hàng gửi phản hồi."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả chi tiết nội dung sự cố, lỗi phát sinh hoặc lý do khiếu nại."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên xử lý của sự cố ('low', 'medium', 'high').",
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}


# ---------------------------------------------------------------------------
# Schema Validator: Kiểm tra tính hợp lệ của tham số trước khi thực thi
# ---------------------------------------------------------------------------

def validate_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Xác thực tham số truyền vào hàm dựa trên JSON Schema của TOOL_DEFINITIONS.
    
    Returns:
        (is_valid, error_message)
    """
    tool_def = next((t for t in TOOL_DEFINITIONS if t["name"] == tool_name), None)
    if not tool_def:
        return False, f"Công cụ '{tool_name}' không tồn tại trong danh mục."

    required_fields = tool_def.get("parameters", {}).get("required", [])
    for field in required_fields:
        if field not in args or args[field] is None:
            return False, f"Thiếu tham số bắt buộc: '{field}' cho công cụ '{tool_name}'."

    properties = tool_def.get("parameters", {}).get("properties", {})
    for field, val in args.items():
        if field in properties and "enum" in properties[field]:
            allowed_enums = properties[field]["enum"]
            if val not in allowed_enums:
                return False, f"Giá trị '{val}' của tham số '{field}' không hợp lệ. Cho phép: {allowed_enums}"

    return True, None
