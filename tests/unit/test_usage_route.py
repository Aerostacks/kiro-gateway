from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from kiro.routes_openai import usage


def _request(account=None):
    manager = MagicMock()
    manager.get_first_account.return_value = account
    state = SimpleNamespace(account_manager=manager, http_client=MagicMock())
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.asyncio
async def test_usage_route_returns_normalized_result():
    account = SimpleNamespace(auth_manager=MagicMock())
    request = _request(account)
    expected = {
        "used": 12.5,
        "limit": 100.0,
        "percent": 12.5,
        "resetsAt": "2026-08-01T00:00:00Z",
        "plan": "Pro",
    }
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(return_value=expected),
    ) as fetch:
        response = await usage(request)

    assert response.status_code == 200
    assert response.body == b'{"used":12.5,"limit":100.0,"percent":12.5,"resetsAt":"2026-08-01T00:00:00Z","plan":"Pro"}'
    fetch.assert_awaited_once_with(account.auth_manager, request.app.state.http_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("account", [None, SimpleNamespace(auth_manager=None)])
async def test_usage_route_requires_initialized_account(account):
    with pytest.raises(HTTPException) as exc:
        await usage(_request(account))
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_usage_route_maps_missing_account_runtime_error_to_unavailable():
    manager = MagicMock()
    manager.get_first_account.side_effect = RuntimeError("No initialized accounts")
    state = SimpleNamespace(account_manager=manager, http_client=MagicMock())
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    with pytest.raises(HTTPException) as exc:
        await usage(request)

    assert exc.value.status_code == 503
    assert exc.value.detail == "No initialized accounts available"


@pytest.mark.asyncio
async def test_usage_route_maps_invalid_payload_to_bad_gateway():
    account = SimpleNamespace(auth_manager=MagicMock())
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=ValueError("missing CREDIT")),
    ):
        with pytest.raises(HTTPException) as exc:
            await usage(_request(account))
    assert exc.value.status_code == 502
    assert exc.value.detail == "Invalid Kiro usage response"


@pytest.mark.asyncio
async def test_usage_route_does_not_leak_upstream_error_details():
    account = SimpleNamespace(auth_manager=MagicMock())
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=RuntimeError("secret upstream detail")),
    ):
        with pytest.raises(HTTPException) as exc:
            await usage(_request(account))
    assert exc.value.status_code == 502
    assert exc.value.detail == "Kiro usage request failed"
