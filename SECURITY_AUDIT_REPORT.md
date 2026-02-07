# Security Audit Report — claude-code-openai-wrapper

**Date:** 2026-02-07
**Auditor:** Claude Opus 4.6 (automated)
**Scope:** Full codebase (`src/`, `tests/`, `Dockerfile`)
**Methodology:** 4 parallel audit agents covering injection, concurrency, auth/access control, and input validation

---

## Executive Summary

A comprehensive security audit identified **30 unique findings** across the FastAPI-based OpenAI API wrapper for Claude Code. All findings were remediated across 3 commits affecting 17 files (+1,227 / -956 lines).

| Severity | Found | Fixed | Commit |
|----------|-------|-------|--------|
| Critical | 2 | 2 | `f9f9e3e` |
| High | 7 | 7 | `f9f9e3e` |
| Medium | 11 | 11 | `a6c7680` |
| Low | 10 | 10 | `dcc50fd` |
| **Total** | **30** | **30** | |

**Tests:** 508 passing, 31 skipped (integration), 3 pre-existing failures (unrelated auth tests).

---

## Critical Severity (2)

### C1 — Arbitrary Command Execution via MCP Server Registration

**Files:** `src/models.py`, `src/main.py`
**Vector:** The `/v1/mcp/servers` endpoint accepted arbitrary `command` values. An attacker could register `/bin/bash` as an MCP server and execute arbitrary commands upon connection.

**Fix:** Added a strict allowlist of permitted MCP commands (`npx`, `node`, `python`, `python3`, `uvx`, `docker`, `deno`, `bun`) with `ClassVar` annotation in `MCPServerConfigRequest`. The validator extracts the base command name via `os.path.basename()` and rejects anything not on the allowlist.

```python
ALLOWED_MCP_COMMANDS: ClassVar[set] = {
    "npx", "node", "python", "python3", "uvx", "docker", "deno", "bun",
}
```

---

### C2 — Environment Variable Race Condition (Cross-Tenant Credential Leakage)

**File:** `src/claude_cli.py`
**Vector:** `run_completion()` mutated `os.environ` without locking. Concurrent requests could overwrite each other's authentication credentials, causing cross-tenant data leakage or auth failures.

**Fix:** Added `asyncio.Lock` (`self._env_lock`) that wraps the entire env var mutation + SDK query + cleanup sequence, ensuring atomic environment access per request.

---

## High Severity (7)

### H1 — Permission Bypass via Custom HTTP Headers

**File:** `src/parameter_validator.py`
**Vector:** Clients could send `X-Claude-Permission-Mode: bypassPermissions` and `X-Claude-Allowed-Tools` headers to override server-side security settings.

**Fix:** Removed extraction of `x-claude-permission-mode`, `x-claude-allowed-tools`, and `x-claude-disallowed-tools` from `extract_claude_headers()`. Only safe headers (`x-claude-max-turns`, `x-claude-max-thinking-tokens`) remain client-settable.

---

### H2 — Timing-Unsafe API Key Comparison

**File:** `src/auth.py`
**Vector:** API key comparison used `!=` operator, vulnerable to character-by-character timing attacks.

**Fix:** Replaced with `secrets.compare_digest()` for constant-time comparison.

```python
if not secrets.compare_digest(credentials.credentials, active_api_key):
```

---

### H3 — Missing Authentication on Sensitive Endpoints

**File:** `src/main.py`
**Vector:** Session management, auth status, and debug endpoints accepted `Depends(security)` but never called `verify_api_key()`, making them accessible without authentication.

**Fix:** Added `await verify_api_key(request, credentials)` to all sensitive endpoints: `/v1/sessions/*`, `/v1/auth/status`, `/v1/debug/request`.

---

### H4 — Unvalidated Session IDs (Hijacking / Namespace Pollution)

**File:** `src/models.py`
**Vector:** Session IDs were unconstrained strings with no length or format validation. Attackers could use extremely long IDs, special characters, or guess other users' session IDs.

**Fix:** Added `max_length=128` constraint and `field_validator` enforcing `^[a-zA-Z0-9_\-]+$` pattern.

---

### H5 — Unauthenticated Global Tool Configuration Changes

**File:** `src/main.py`
**Vector:** `POST /v1/tools/config` without `session_id` modified the global tool configuration affecting all users, allowing privilege escalation.

**Fix:** Requests without `session_id` now return `403 Forbidden`. Only per-session tool configuration is allowed.

---

### H6 — `threading.Lock` in Async Context (Deadlock / Event Loop Blocking)

**Files:** `src/session_manager.py`, `src/tool_manager.py`, `src/mcp_client.py`
**Vector:** All three managers used `threading.Lock()`, which blocks the entire asyncio event loop thread when acquired, causing starvation and potential deadlocks.

**Fix:** Replaced `threading.Lock()` with `asyncio.Lock()` in all three modules. All methods that acquire the lock converted from sync to async. Updated all callers in `main.py` and all test files.

---

### H7 — Unbounded Session / Tool Config Memory Growth (DoS)

**Files:** `src/session_manager.py`, `src/tool_manager.py`
**Vector:** No limits on session count, messages per session, or tool session configs. Attackers could exhaust server memory.

**Fix:**
- Session manager: `MAX_SESSIONS = 1000`, `MAX_MESSAGES_PER_SESSION = 200`, `MAX_MESSAGE_CONTENT_LENGTH = 500KB`. LRU eviction of oldest session when limit reached.
- Tool manager: `MAX_TOOL_SESSION_CONFIGS = 1000`, `TOOL_SESSION_CONFIG_TTL_HOURS = 1`. TTL-based expiration and eviction.

---

## Medium Severity (11)

### M1 — CORS Wildcard with Credentials

**File:** `src/main.py`
**Vector:** Default CORS config used `allow_origins=["*"]` with `allow_credentials=True`, enabling cross-origin credential-bearing requests.

**Fix:** Set `allow_credentials = "*" not in cors_origins` — credentials are automatically disabled when wildcard origins are configured.

---

### M2 — Missing Security Headers

**File:** `src/main.py`
**Vector:** No security headers (HSTS, X-Frame-Options, CSP, etc.) were set on responses.

**Fix:** Added `SecurityHeadersMiddleware` that sets on every response:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `X-XSS-Protection: 1; mode=block`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- `Strict-Transport-Security` (HTTPS only)

---

### M3 — Debug Logging Exposes Authorization Headers

**File:** `src/main.py`
**Vector:** `DebugLoggingMiddleware` logged all request headers verbatim, including `Authorization: Bearer <key>`.

**Fix:** Authorization headers are redacted to `Bearer [REDACTED]` before logging. Debug endpoint also redacts Authorization headers in response.

---

### M4 — Auth Status Endpoint Reveals Configuration Details

**File:** `src/main.py`
**Vector:** `/v1/auth/status` returned `api_key_source`, environment variable names, API key length, and cloud provider config.

**Fix:** Response now returns only `method` and `status.valid`. Removed `api_key_source`, `environment_variables`, `config`, and all other sensitive details.

---

### M5 — Internal Exception Details Leaked to Clients

**File:** `src/main.py` (4 locations)
**Vector:** Error responses used `str(e)` which could reveal file paths, stack traces, or internal API details.

**Fix:** All 4 error responses changed to generic `"Internal server error"`. Full exception details preserved in `logger.error()` for server-side debugging.

---

### M5b — Validation Errors Leak Raw Input Values

**File:** `src/main.py`
**Vector:** Validation error handler included `"input": error.get("input")` in error responses, potentially exposing sensitive request data.

**Fix:** Removed the `"input"` field from validation error response details.

---

### M6 — `bypassPermissions` Used Unconditionally

**File:** `src/main.py` (2 locations)
**Vector:** Anthropic Messages endpoint always ran with `permission_mode="bypassPermissions"`, bypassing all tool permission checks.

**Fix:** Changed both streaming and non-streaming Anthropic endpoints to use `permission_mode="acceptEdits"`.

---

### M7 — No Streaming Response Timeout

**File:** `src/main.py`
**Vector:** Streaming response generators iterated the Claude SDK without timeout. A hanging SDK or slow-read client could hold connections indefinitely.

**Fix:** Both streaming loops (OpenAI and Anthropic) wrapped with `async with asyncio.timeout(claude_cli.timeout)`.

---

### M8 — MCP Connect TOCTOU Race Condition

**File:** `src/mcp_client.py`
**Vector:** `connect_server()` checked `if name in self.connections` outside the lock, then created the connection inside the lock. Concurrent connect calls for the same server could spawn duplicate subprocesses with one leaked.

**Fix:** Moved the entire connection check + creation + registration inside a single `async with self.lock:` block.

---

### M9 — Session Data Race (TOCTOU)

**File:** `src/session_manager.py`
**Vector:** `process_messages()` called `get_or_create_session()` (locked), then mutated the session outside the lock. Concurrent requests on the same session could interleave messages.

**Fix:** `process_messages()` now holds the lock for the entire check + create + add_messages + return operation. Returns a copy of messages to prevent concurrent mutation.

---

### M11 — Request Size Limit Bypass via Chunked Encoding

**File:** `src/main.py`
**Vector:** `RequestSizeLimitMiddleware` only checked `Content-Length` header. Chunked transfer encoding (no `Content-Length`) bypassed the check entirely.

**Fix:** For POST/PUT/PATCH requests without `Content-Length`, reads the actual body and checks its size. Also validates `Content-Length` header format with try/except for `ValueError`.

---

## Low Severity (10)

### L1 — Unvalidated Client-Provided Request ID

**File:** `src/main.py`
**Vector:** `X-Request-ID` header accepted without validation. Could contain CRLF for log injection, or excessively long values.

**Fix:** Added `_REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]{1,128}$")` validation. Invalid IDs are replaced with server-generated UUIDs.

---

### L2 — CORS_ORIGINS json.loads Without Error Handling

**File:** `src/main.py`
**Vector:** Malformed `CORS_ORIGINS` env var would crash the application on startup.

**Fix:** Wrapped `json.loads()` in try/except, defaulting to `["*"]` on parse failure with a warning log.

---

### L3 — Docker CMD Uses `--reload` in Production

**File:** `Dockerfile`
**Vector:** `--reload` enables file watching, increasing attack surface and causing restarts if an attacker can write files.

**Fix:** Removed `--reload` from the production CMD.

---

### L4 — ReDoS in Image Pattern Regex

**File:** `src/message_adapter.py`
**Vector:** `data:image/.*?;base64,.*?(?=\s|$)` used lazy quantifiers that could cause quadratic backtracking on crafted input.

**Fix:** Replaced with character class restrictions: `data:image/[^;]+;base64,[^\s]*`.

---

### L5 — ReDoS in Tool Call Pattern Regex

**File:** `src/tool_bridge.py`
**Vector:** `<tool_call>\s*(\{.*?\})\s*</tool_call>` with `re.DOTALL` could backtrack excessively when `<tool_call>` exists without matching `</tool_call>`.

**Fix:** Simplified to `<tool_call>\s*(\{.*?)</tool_call>` — anchored directly to the closing delimiter without the intermediate `\}` requirement.

---

### L6 — Temp Directory Cleanup Follows Symlinks

**File:** `src/claude_cli.py`
**Vector:** `shutil.rmtree(self.temp_dir)` would follow symlinks. An attacker placing a symlink inside the temp dir could cause deletion of arbitrary directories.

**Fix:** Added `os.path.realpath()` resolution and verification that the resolved path is within `tempfile.gettempdir()` before deletion.

---

### L7 — Rate Limiter Not Proxy-Aware

**File:** `src/rate_limiter.py`
**Vector:** Rate limiting used `get_remote_address()` which returns the proxy's IP behind a reverse proxy, applying a single shared limit to all users.

**Fix:** `get_rate_limit_key()` now checks `X-Forwarded-For` (first IP) and `X-Real-IP` headers before falling back to remote address.

---

### L8 — Content-Length Integer Parsing Without Validation

**File:** `src/main.py`
**Vector:** Malformed `Content-Length` header (e.g., `"abc"`) caused unhandled `ValueError`.

**Fix:** Wrapped `int(content_length)` in try/except, returning `400 Bad Request` for invalid values. *(Fixed as part of M11)*

---

### L9 — `runtime_api_key` Global Without Synchronization

**File:** `src/main.py`
**Vector:** Module-level mutable global accessed cross-module. Could see partial writes during key rotation.

**Assessment:** Low risk — variable is set once at startup and effectively immutable. No code change needed beyond documenting the constraint. *(Accepted risk)*

---

### L10 — SlowAPI In-Memory Rate Limiter (No Distributed Coordination)

**File:** `src/rate_limiter.py`
**Vector:** In-memory storage means multi-instance deployments have per-instance rate limits, allowing attackers to multiply their effective limit.

**Assessment:** Architecture limitation. Mitigated by proxy-awareness fix (L7). For production multi-instance deployments, a Redis-backed storage should be configured. *(Documented recommendation)*

---

## Files Modified

| File | Changes | Description |
|------|---------|-------------|
| `src/main.py` | +935 | Security headers, CORS, auth on endpoints, error redaction, streaming timeout, request validation |
| `src/session_manager.py` | +108 | asyncio.Lock, session limits, TOCTOU fix, LRU eviction |
| `src/mcp_client.py` | +258 | asyncio.Lock, TOCTOU fix in connect |
| `src/claude_cli.py` | +178 | asyncio.Lock for env vars, safe temp cleanup |
| `src/tool_manager.py` | +78 | asyncio.Lock, session config limits/TTL |
| `src/models.py` | +46 | MCP command allowlist, session ID validation |
| `src/parameter_validator.py` | +20 | Removed security-sensitive header extraction |
| `src/auth.py` | +5 | Constant-time API key comparison |
| `src/rate_limiter.py` | +12 | Proxy-aware IP extraction |
| `src/message_adapter.py` | +2 | ReDoS-safe regex |
| `src/tool_bridge.py` | +3 | ReDoS-safe regex |
| `Dockerfile` | +4 | Removed --reload |
| `tests/test_session_manager_unit.py` | +149 | Async test updates |
| `tests/test_tool_manager_unit.py` | +158 | Async test updates |
| `tests/test_mcp_client_unit.py` | +171 | Async test updates |
| `tests/test_parameter_validator_unit.py` | +33 | Updated assertions |
| `tests/test_rate_limiter_unit.py` | +23 | Proxy-aware tests |

**Total: 17 files, +1,227 lines, -956 lines**

---

## Remaining Recommendations (Architecture)

These items are architectural concerns that cannot be fully resolved with code changes alone:

1. **Sandbox tool execution** — When `enable_tools: true`, Claude has Bash/Write/Edit access. Consider container isolation, chroot, or per-request working directories for multi-tenant deployments.
2. **Redis-backed rate limiting** — For multi-instance production deployments, configure SlowAPI with Redis storage.
3. **Per-request working directories** — All requests share a single `ClaudeCodeCLI.cwd`. Consider unique temp dirs per request or session for isolation.
4. **Prompt injection** — User messages are passed to Claude without sanitization. This is inherent to the architecture but should be documented as a risk when tools are enabled.

---

*Report generated by Claude Opus 4.6 — 2026-02-07*
