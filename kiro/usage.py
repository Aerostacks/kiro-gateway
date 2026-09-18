"""Fetch and normalize Kiro account-level monthly credit usage."""

from datetime import datetime, timezone
import math
from typing import Any, Iterable

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


def aggregate_credit_usage(
    usages: Iterable[dict[str, Any]],
    *,
    total_accounts: int,
) -> dict[str, Any]:
    """Sum valid account windows without fabricating shared plan/reset metadata."""
    usage_list = list(usages)
    if not usage_list:
        raise ValueError("No valid account usage responses")
    if total_accounts < len(usage_list) or total_accounts <= 0:
        raise ValueError("Invalid total account count")

    used_values: list[float] = []
    limit_values: list[float] = []
    for usage in usage_list:
        try:
            used = float(usage["used"])
            limit = float(usage["limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Account usage has nonnumeric values") from exc
        if not math.isfinite(used) or not math.isfinite(limit) or used < 0 or limit <= 0:
            raise ValueError("Account usage has invalid values")
        used_values.append(used)
        limit_values.append(limit)

    try:
        used_total = math.fsum(used_values)
        limit_total = math.fsum(limit_values)
    except OverflowError as exc:
        raise ValueError("Account usage totals overflow") from exc
    if not math.isfinite(used_total) or not math.isfinite(limit_total):
        raise ValueError("Account usage totals overflow")

    plans = sorted({
        usage["plan"]
        for usage in usage_list
        if isinstance(usage.get("plan"), str) and usage["plan"]
    })
    all_plans_known = all(
        isinstance(usage.get("plan"), str) and bool(usage["plan"])
        for usage in usage_list
    )
    reset_dates = {usage.get("resetsAt") for usage in usage_list}
    mixed_reset_dates = len(reset_dates) > 1
    successful_accounts = len(usage_list)
    failed_accounts = total_accounts - successful_accounts

    return {
        "used": used_total,
        "limit": limit_total,
        "percent": round(
            min(100.0, max(0.0, used_total / limit_total * 100.0)),
            6,
        ),
        "resetsAt": next(iter(reset_dates)) if not mixed_reset_dates else None,
        "plan": (
            plans[0]
            if all_plans_known and len(plans) == 1
            else ("Multiple plans" if all_plans_known and plans else None)
        ),
        "plans": plans,
        "accountCount": total_accounts,
        "successfulAccountCount": successful_accounts,
        "failedAccountCount": failed_accounts,
        "partial": failed_accounts > 0,
        "mixedResetDates": mixed_reset_dates,
    }


async def fetch_credit_usage(auth_manager, client: httpx.AsyncClient) -> dict[str, Any]:
    """Call Kiro's GetUsageLimits operation, refreshing once on auth failure."""
    payload = {
        "origin": "KIRO_CLI",
        "resourceType": "AGENTIC_REQUEST",
        "isEmailRequired": False,
        "profileArn": auth_manager.profile_arn,
    }
    url = auth_manager.usage_host

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
