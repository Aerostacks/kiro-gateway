import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from kiro.routes_openai import usage


def _request(account=None, accounts=None):
    manager = MagicMock()
    manager.get_first_account.return_value = account
    account_system = accounts is not None
    if account_system:
        manager.get_accounts_for_usage = AsyncMock(return_value=accounts)
    state = SimpleNamespace(
        account_manager=manager,
        account_system=account_system,
        http_client=MagicMock(),
    )
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
    body = response.body.decode()
    assert '"accountCount":1' in body
    assert '"id":"account-1"' in body
    assert '"available":true' in body
    fetch.assert_awaited_once_with(account.auth_manager, request.app.state.http_client)


@pytest.mark.asyncio
async def test_usage_route_aggregates_every_enabled_account():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=MagicMock()),
    ]
    request = _request(accounts=accounts)
    usages = [
        {
            "used": 10.0, "limit": 100.0, "percent": 10.0,
            "resetsAt": "2026-08-01T00:00:00Z", "plan": "Pro",
        },
        {
            "used": 20.0, "limit": 100.0, "percent": 20.0,
            "resetsAt": "2026-08-01T00:00:00Z", "plan": "Pro",
        },
    ]
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=usages),
    ) as fetch:
        response = await usage(request)

    body = response.body.decode()
    assert '"used":30.0' in body
    assert '"limit":200.0' in body
    assert '"accountCount":2' in body
    assert '"successfulAccountCount":2' in body
    assert '"id":"account-1"' in body
    assert '"id":"account-2"' in body
    assert body.count('"available":true') == 2
    assert fetch.await_count == 2
    request.app.state.account_manager.get_accounts_for_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_usage_route_returns_partial_total_when_one_account_fails():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=MagicMock()),
    ]
    request = _request(accounts=accounts)
    valid = {
        "used": 10.0, "limit": 100.0, "percent": 10.0,
        "resetsAt": "2026-08-01T00:00:00Z", "plan": "Pro",
    }
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=[valid, RuntimeError("upstream unavailable")]),
    ):
        response = await usage(request)

    body = response.body.decode()
    assert '"partial":true' in body
    assert '"successfulAccountCount":1' in body
    assert '"failedAccountCount":1' in body
    assert '"id":"account-2","available":false' in body


@pytest.mark.asyncio
async def test_usage_route_includes_uninitialized_account_as_unavailable():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=None),
    ]
    request = _request(accounts=accounts)
    valid = {
        "used": 10.0, "limit": 100.0, "percent": 10.0,
        "resetsAt": "2026-08-01T00:00:00Z", "plan": "Pro",
    }
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(return_value=valid),
    ) as fetch:
        response = await usage(request)

    body = response.body.decode()
    assert '"successfulAccountCount":1' in body
    assert '"failedAccountCount":1' in body
    assert '"id":"account-2","available":false' in body
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_usage_route_propagates_account_fetch_cancellation():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=MagicMock()),
    ]
    request = _request(accounts=accounts)
    valid = {
        "used": 10.0, "limit": 100.0, "percent": 10.0,
        "resetsAt": "2026-08-01T00:00:00Z", "plan": "Pro",
    }
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=[valid, asyncio.CancelledError()]),
    ):
        with pytest.raises(asyncio.CancelledError):
            await usage(request)


@pytest.mark.asyncio
async def test_usage_route_propagates_child_cancellation_without_waiting_for_sibling():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=MagicMock()),
    ]
    request = _request(accounts=accounts)
    both_started = asyncio.Event()
    release_cancellation = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    started = 0

    async def fetch(auth_manager, _client):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        if auth_manager is accounts[0].auth_manager:
            await release_cancellation.wait()
            raise asyncio.CancelledError
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    with patch("kiro.routes_openai.fetch_credit_usage", side_effect=fetch):
        route_task = asyncio.create_task(usage(request))
        await both_started.wait()
        release_cancellation.set()
        done, _ = await asyncio.wait({route_task}, timeout=0.5)
        try:
            assert route_task in done
            with pytest.raises(asyncio.CancelledError):
                route_task.result()
            assert sibling_cancelled.is_set()
        finally:
            if not route_task.done():
                route_task.cancel()
                await asyncio.gather(route_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_usage_route_maps_aggregate_overflow_to_bad_gateway():
    accounts = [
        SimpleNamespace(auth_manager=MagicMock()),
        SimpleNamespace(auth_manager=MagicMock()),
    ]
    request = _request(accounts=accounts)
    overflowing = {
        "used": 1e308,
        "limit": 1e308,
        "percent": 100.0,
        "resetsAt": None,
        "plan": "Pro",
    }
    with patch(
        "kiro.routes_openai.fetch_credit_usage",
        new=AsyncMock(side_effect=[overflowing, overflowing]),
    ):
        with pytest.raises(HTTPException) as exc:
            await usage(request)

    assert exc.value.status_code == 502
    assert exc.value.detail == "Invalid Kiro usage response"


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
