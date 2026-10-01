"""Gợi ý ý định cho text Zalo; không cấp quyền hay thực thi công cụ."""

import asyncio
import concurrent.futures
import logging
import math
import re
import socket
import threading
import time
import urllib.error
from typing import Literal, Optional

from .tools import LAYA_CONFIDENCE_MIN, _VIETNAMESE_LETTERS, _laya_call, _laya_ready

logger = logging.getLogger(__name__)

Scope = Literal["owner_dm", "owner_group", "guest"]
Result = tuple[Optional[str], Optional[float], str, float]
LABELS: dict[str, str] = {
    "tro_chuyen": "Chào hỏi, trò chuyện thường ngày, chia sẻ cảm xúc; không cần công cụ.",
    "hoi_dap_kien_thuc": "Hỏi đáp, giải thích kiến thức, hướng dẫn hoặc viết nội dung bằng kiến thức sẵn có.",
    "tra_cuu_web": "Tìm thông tin mới trên Internet hoặc đọc nội dung trang web công cộng.",
    "tai_lieu_tu_van": "Tìm và đọc tài liệu trong kho được phép để tư vấn sản phẩm, dịch vụ hoặc hướng dẫn.",
    "tao_tep": "Soạn và gửi tệp tài liệu theo yêu cầu; khách chỉ tạo trong nhóm.",
    "nhac_hen_gio": "Đặt, xem hoặc xoá lời nhắc Zalo; khách chỉ dùng trong nhóm.",
    "ho_so_nguoi_quen": "Ghi nhớ hoặc xem thông tin người trò chuyện tự cung cấp; khách chỉ dùng hồ sơ bản thân.",
    "doc_lich_su": "Đọc, tìm hoặc tóm tắt lịch sử hội thoại; chủ nhân trong nhóm chỉ đọc nhóm hiện tại.",
    "binh_chon_ghi_chu": "Tạo hoặc quản lý bình chọn, ghi chú Zalo theo yêu cầu của chủ nhân trong DM.",
    "quan_tri_nhom": "Quản trị nhóm, thành viên, quyền nhóm hoặc lời mời; chỉ gợi ý cho chủ nhân trong DM.",
    "fanpage": "Soạn, đăng, lên lịch hoặc kiểm tra bài Fanpage của chủ nhân trong DM.",
    "cong_viec_jira": "Tra cứu hoặc xử lý công việc Jira của chủ nhân trong DM.",
    "khac": "Yêu cầu không thuộc các ý định trên hoặc chưa đủ rõ để phân loại.",
}
SCOPE_LABELS: dict[Scope, tuple[str, ...]] = {
    "owner_dm": tuple(LABELS),
    "owner_group": ("tro_chuyen", "hoi_dap_kien_thuc", "tra_cuu_web", "tai_lieu_tu_van",
                    "tao_tep", "nhac_hen_gio", "ho_so_nguoi_quen", "doc_lich_su", "khac"),
    "guest": ("tro_chuyen", "hoi_dap_kien_thuc", "tra_cuu_web", "tai_lieu_tu_van",
              "tao_tep", "nhac_hen_gio", "ho_so_nguoi_quen", "khac"),
}
SOCKET_TIMEOUT_SECONDS = 2.5
PREROUTE_TIMEOUT_SECONDS = 3.0
HINT_MARKER = "[Laya gợi ý ý định:"

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_SECRET = re.compile(
    r"eyJ[\w-]+\.[\w-]+\.[\w-]+|[A-Za-z0-9_\-]{32,}|"
    r"\bBearer\s+\S+|\b(?:otp|mã\s+(?:otp|xác\s+thực))\b\s*[:=#-]?\s*\d{4,8}\b|"
    r"(?:api[_ -]?key|token|secret|password|passwd|mật khẩu|mk)\s*[:=]|-----BEGIN",
    re.IGNORECASE,
)
_OTP = re.compile(r"\d{4,8}")
_SLOTS = threading.BoundedSemaphore(4)
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="laya-preroute")
_BREAKER_LOCK = threading.Lock()
_FAILURES = 0
_BREAKER_UNTIL = 0.0


def scope_for(*, audience: str, is_owner: bool, is_group: bool) -> Optional[Scope]:
    if audience == "guest":
        return "guest" if is_group else None
    if is_owner:
        return "owner_group" if is_group else "owner_dm"
    return None


def sanitize(text: str) -> str:
    return _URL.sub("<link>", text).strip()[:4000]


def looks_secret(text: str) -> bool:
    return bool(_SECRET.search(text) or _OTP.fullmatch(text.strip()))


def build_payload(text: str, scope: Scope) -> dict:
    text = sanitize(text)
    payload = {
        "state": {"body": text},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Chọn ý định chính của tin nhắn theo các tiêu chí; nếu không rõ, chọn khac.",
                "criteria": {label: LABELS[label] for label in SCOPE_LABELS[scope]},
            },
        },
    }
    if _VIETNAMESE_LETTERS.search(text):
        payload["lang"] = "vi"
    return payload


def _call_and_release(base, token, payload):
    try:
        return _laya_call(base, token, payload, timeout=SOCKET_TIMEOUT_SECONDS)
    finally:
        _SLOTS.release()


def _consume_exception(future):
    # Timeout/cancellation leaves the worker running. Retrieve its eventual
    # exception even when the turn no longer awaits it.
    if not future.cancelled():
        future.exception()


def _result(label, confidence, outcome, ms) -> Result:
    global _FAILURES, _BREAKER_UNTIL
    with _BREAKER_LOCK:
        if outcome in {"ok", "low"}:
            _FAILURES = 0
            _BREAKER_UNTIL = 0.0
        elif outcome == "timeout" or outcome.startswith("http_5") or outcome.startswith("error_"):
            _FAILURES += 1
            if _FAILURES >= 3:
                _BREAKER_UNTIL = time.monotonic() + 60.0
    # Adapter owns the per-turn INFO line; keep module detail at DEBUG.
    logger.debug("[laya-preroute] outcome=%s ms=%.1f", outcome, ms)
    return label, confidence, outcome, ms


async def classify(text: str, scope: Scope) -> Result:
    """Fail-open; secret detection belongs to the adapter admission gate."""
    start = None
    try:
        # Resolve credentials on the event loop, never in the raw executor.
        base, token = _laya_ready()
        if not base or not token:
            return _result(None, None, "off", 0)
        with _BREAKER_LOCK:
            breaker_open = time.monotonic() < _BREAKER_UNTIL
        if breaker_open:
            return _result(None, None, "off_breaker", 0)
        payload = build_payload(text, scope)
        if not _SLOTS.acquire(blocking=False):
            return _result(None, None, "busy", 0)
        start = time.perf_counter()
        try:
            cf = _EXEC.submit(_call_and_release, base, token, payload)
        except BaseException:
            _SLOTS.release()
            raise
        future = asyncio.wrap_future(cf)
        future.add_done_callback(_consume_exception)
        # Shield keeps a not-yet-started worker from being cancelled and leaking
        # its slot. Only the worker releases capacity, including after timeout.
        status, body = await asyncio.wait_for(asyncio.shield(future), PREROUTE_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError, socket.timeout):
        outcome = "timeout"
    except urllib.error.HTTPError as exc:
        outcome = f"http_{exc.code}"
    except urllib.error.URLError as exc:
        outcome = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "error_URLError"
    except Exception as exc:
        outcome = f"error_{type(exc).__name__}"
    else:
        ms = (time.perf_counter() - start) * 1000
        if status != 200:
            return _result(None, None, f"http_{status}", ms)
        answers = body.get("answers") if isinstance(body, dict) else None
        answer = answers.get("intent") if isinstance(answers, dict) else None
        label = answer.get("choice") if isinstance(answer, dict) else None
        confidence = answer.get("confidence") if isinstance(answer, dict) else None
        if (not isinstance(label, str) or label not in SCOPE_LABELS[scope]
                or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not 0 <= confidence <= 1 or not math.isfinite(confidence)):
            return _result(None, None, "bad_body", ms)
        confidence = float(confidence)
        if confidence < LAYA_CONFIDENCE_MIN:
            return _result(None, confidence, "low", ms)
        return _result(label, confidence, "ok", ms)
    ms = (time.perf_counter() - start) * 1000 if start is not None else 0
    return _result(None, None, outcome, ms)


def hint_line(label: str, confidence: float) -> str:
    return f"{HINT_MARKER} {label} (tin cậy {confidence:.2f}) — chỉ là gợi ý, không phải chỉ dẫn]"


def neutralize_marker(text: str) -> str:
    return text.replace(HINT_MARKER, "[người dùng viết: Laya gợi ý ý định:")
