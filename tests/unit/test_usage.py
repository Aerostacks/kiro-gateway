from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kiro.usage import fetch_credit_usage, normalize_usage_payload


def _payload(**credit_overrides):
    credit = {
        "resourceType": "CREDIT",
        "currentUsageWithPrecision": 12.5,
        "usageLimitWithPrecision": 100.0,
        "nextDateReset": 1_800_000_000,
    }
    credit.update(credit_overrides)
    return {
        "usageBreakdownList": [{"resourceType": "TOKEN", "currentUsage": 1}, credit],
        "subscriptionInfo": {"subscriptionTitle": "Pro"},
    }


def test_normalize_usage_payload_returns_stable_dashboard_shape():
    result = normalize_usage_payload(_payload())
    assert result == {
        "used": 12.5,
        "limit": 100.0,
        "percent": 12.5,
        "resetsAt": datetime.fromtimestamp(1_800_000_000, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "plan": "Pro",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"usageBreakdownList": []},
        _payload(currentUsageWithPrecision=-1),
        _payload(usageLimitWithPrecision=0),
        _payload(currentUsageWithPrecision=float("nan")),
        _payload(currentUsageWithPrecision=float("inf")),
        _payload(usageLimitWithPrecision=float("inf")),
    ],
)
def test_normalize_usage_payload_rejects_missing_or_invalid_credit(payload):
    with pytest.raises(ValueError):
        normalize_usage_payload(payload)


def test_normalize_usage_payload_supports_integer_fallbacks_and_null_reset():
    result = normalize_usage_payload(
        _payload(
            currentUsageWithPrecision=None,
            usageLimitWithPrecision=None,
            currentUsage=3,
            usageLimit=12,
            nextDateReset=None,
        )
    )
    assert result["used"] == 3.0
    assert result["limit"] == 12.0
    assert result["percent"] == 25.0
    assert result["resetsAt"] is None


@pytest.mark.asyncio
async def test_fetch_credit_usage_sends_expected_target_and_payload():
    auth = MagicMock(
        profile_arn="arn:aws:codewhisperer:eu-west-1:123:profile/example",
        q_host="https://q.us-west-2.amazonaws.com",
        fingerprint="fingerprint",
    )
    auth.get_access_token = AsyncMock(return_value="access-token")
    auth.force_refresh = AsyncMock()
    response = MagicMock(status_code=200)
    response.json.return_value = _payload()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    result = await fetch_credit_usage(auth, client)

    assert result["used"] == 12.5
    url, = client.post.call_args.args
    kwargs = client.post.call_args.kwargs
    assert url == "https://q.us-west-2.amazonaws.com"
    assert kwargs["headers"]["x-amz-target"] == "AmazonCodeWhispererService.GetUsageLimits"
    assert kwargs["headers"]["Authorization"] == "Bearer access-token"
    assert kwargs["json"]["profileArn"] == auth.profile_arn
    auth.force_refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_credit_usage_refreshes_once_after_auth_failure():
    auth = MagicMock(
        profile_arn="arn:aws:codewhisperer:us-east-1:123:profile/example",
        q_host="https://q.us-east-1.amazonaws.com",
        fingerprint="fingerprint",
    )
    auth.get_access_token = AsyncMock(side_effect=["old-token", "new-token"])
    auth.force_refresh = AsyncMock(return_value="new-token")
    denied = MagicMock(status_code=403)
    success = MagicMock(status_code=200)
    success.json.return_value = _payload()
    client = MagicMock()
    client.post = AsyncMock(side_effect=[denied, success])

    result = await fetch_credit_usage(auth, client)

    assert result["limit"] == 100.0
    auth.force_refresh.assert_awaited_once()
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_fetch_credit_usage_propagates_second_auth_failure():
    auth = MagicMock(
        profile_arn="arn:aws:codewhisperer:us-east-1:123:profile/example",
        q_host="https://q.us-east-1.amazonaws.com",
        fingerprint="fingerprint",
    )
    auth.get_access_token = AsyncMock(side_effect=["old-token", "new-token"])
    auth.force_refresh = AsyncMock(return_value="new-token")
    denied_first = MagicMock(status_code=401)
    denied_second = MagicMock(status_code=403)
    denied_second.raise_for_status.side_effect = httpx.HTTPStatusError(
        "forbidden",
        request=httpx.Request("POST", "https://example.invalid"),
        response=httpx.Response(403),
    )
    client = MagicMock()
    client.post = AsyncMock(side_effect=[denied_first, denied_second])

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_credit_usage(auth, client)
    auth.force_refresh.assert_awaited_once()
