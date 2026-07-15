"""Fetch and normalize Kiro account-level monthly credit usage."""

from datetime import datetime, timezone
import math
from typing import Any

import httpx

from kiro.utils import get_kiro_headers


def normalize_usage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the monthly CREDIT window in a stable dashboard-friendly shape."""
    breakdowns = payload.get("usageBreakdownList") or []
    credit = next(
        (item for item in breakdowns if item.get("resourceType") == "CREDIT"),
        None,
    )
    if credit is None:
        raise ValueError("Kiro usage response has no CREDIT breakdown")

    used_value = credit.get("currentUsageWithPrecision")
    if used_value is None:
        used_value = credit.get("currentUsage", 0)
    limit_value = credit.get("usageLimitWithPrecision")
    if limit_value is None:
        limit_value = credit.get("usageLimit", 0)
    try:
        used = float(used_value)
        limit = float(limit_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kiro CREDIT breakdown has nonnumeric usage values") from exc
    if not math.isfinite(used) or not math.isfinite(limit) or used < 0 or limit <= 0:
        raise ValueError("Kiro CREDIT breakdown has invalid usage values")

    reset_timestamp = credit.get("nextDateReset", payload.get("nextDateReset"))
    resets_at = None
    if reset_timestamp is not None:
        try:
            reset_value = float(reset_timestamp)
            if not math.isfinite(reset_value):
                raise ValueError
            resets_at = (
                datetime.fromtimestamp(reset_value, timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise ValueError("Kiro CREDIT breakdown has invalid reset timestamp") from exc

    subscription = payload.get("subscriptionInfo") or {}
    return {
        "used": used,
        "limit": limit,
        "percent": round(min(100.0, max(0.0, used / limit * 100.0)), 6),
        "resetsAt": resets_at,
        "plan": subscription.get("subscriptionTitle"),
    }


async def fetch_credit_usage(auth_manager, client: httpx.AsyncClient) -> dict[str, Any]:
    """Call Kiro's GetUsageLimits operation, refreshing once on auth failure."""
    payload = {
        "origin": "KIRO_CLI",
        "resourceType": "AGENTIC_REQUEST",
        "isEmailRequired": False,
        "profileArn": auth_manager.profile_arn,
    }
    url = auth_manager.q_host

    for attempt in range(2):
        token = await auth_manager.get_access_token()
        headers = get_kiro_headers(auth_manager, token)
        headers["x-amz-target"] = "AmazonCodeWhispererService.GetUsageLimits"
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return normalize_usage_payload(response.json())
        if response.status_code in (401, 403) and attempt == 0:
            await auth_manager.force_refresh()
            continue
        response.raise_for_status()

    raise RuntimeError("Kiro usage request failed after token refresh")
