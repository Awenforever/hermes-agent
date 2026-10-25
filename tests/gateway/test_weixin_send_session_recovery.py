"""Regression: the Weixin send session must recycle after a transport failure.

A long-lived aiohttp send session can wedge after a network blip or a proxy
reset: every later send fails with ``Cannot connect to host ... ssl:default``
while a fresh session to the same host succeeds, so the gateway only recovers
on restart. The poll loop recycles its session after a failure streak
(#79889); sends never did. These tests pin the send-side recycle.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter as weixin
from plugins.platforms.weixin.adapter import WeixinAdapter


def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={"account_id": "test-account"})
    )


def _session() -> MagicMock:
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    return session


class TestTransportFailureDetection(unittest.TestCase):
    def test_connection_errors_are_transport_failures(self):
        self.assertTrue(
            weixin._is_transport_failure(
                ConnectionError("Cannot connect to host ilinkai.weixin.qq.com:443 ssl:default [None]")
            )
        )
        self.assertTrue(weixin._is_transport_failure(TimeoutError()))
        if weixin.aiohttp is not None:
            self.assertTrue(
                weixin._is_transport_failure(weixin.aiohttp.ClientConnectionError("Cannot connect"))
            )

    def test_wrapped_connection_error_is_a_transport_failure(self):
        try:
            try:
                raise ConnectionError("Cannot connect to host ilinkai.weixin.qq.com:443 ssl:default [None]")
            except ConnectionError as cause:
                raise RuntimeError("send failed") from cause
        except RuntimeError as exc:
            self.assertTrue(weixin._is_transport_failure(exc))

    def test_api_and_value_errors_keep_the_session(self):
        self.assertFalse(weixin._is_transport_failure(RuntimeError("iLink sendmessage error: ret=-2 errcode=-2")))
        self.assertFalse(weixin._is_transport_failure(ValueError("text must not be empty")))


class TestRecycleSendSession(unittest.TestCase):
    def test_recycle_replaces_then_closes_the_old_session(self):
        adapter = _make_adapter()
        adapter._running = True
        old, fresh = _session(), _session()
        adapter._send_session = old

        with patch.object(weixin, "_new_session", return_value=fresh) as mk:
            asyncio.run(adapter._recycle_send_session())

        mk.assert_called_once()
        self.assertIs(adapter._send_session, fresh)
        old.close.assert_awaited_once()

    def test_stale_session_race_keeps_the_newer_session(self):
        adapter = _make_adapter()
        adapter._running = True
        current, stale = _session(), _session()
        adapter._send_session = current

        with patch.object(weixin, "_new_session") as mk:
            asyncio.run(adapter._recycle_send_session(stale=stale))

        mk.assert_not_called()
        self.assertIs(adapter._send_session, current)


class TestSendRecyclesOnTransportFailure(unittest.TestCase):
    def _ready_adapter(self):
        adapter = _make_adapter()
        adapter._running = True
        adapter._send_chunk_retries = 1
        adapter._send_chunk_retry_delay_seconds = 0.0
        adapter._rate_limit_circuit_threshold = 99
        return adapter

    def test_text_send_retries_on_a_fresh_session_after_transport_failure(self):
        adapter = self._ready_adapter()
        old, fresh = _session(), _session()
        adapter._send_session = old
        seen = []

        async def _fake_send_message(session, **kwargs):
            seen.append(session)
            if session is old:
                raise ConnectionError("Cannot connect to host ilinkai.weixin.qq.com:443 ssl:default [None]")
            return {"ret": 0, "errcode": 0}

        with patch.object(weixin, "_new_session", return_value=fresh), \
                patch.object(weixin, "_send_message", _fake_send_message):
            asyncio.run(
                adapter._send_text_chunk(chat_id="chat", chunk="hi", context_token=None, client_id="cid")
            )

        self.assertEqual(seen, [old, fresh])
        self.assertIs(adapter._send_session, fresh)
        old.close.assert_awaited_once()

    def test_api_error_does_not_recycle_the_session(self):
        adapter = self._ready_adapter()
        old = _session()
        adapter._send_session = old

        async def _fake_send_message(session, **kwargs):
            return {"ret": -3, "errcode": -3, "errmsg": "bad request"}

        with patch.object(weixin, "_new_session") as mk, \
                patch.object(weixin, "_send_message", _fake_send_message):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    adapter._send_text_chunk(chat_id="chat", chunk="hi", context_token=None, client_id="cid")
                )

        mk.assert_not_called()
        self.assertIs(adapter._send_session, old)

    def test_media_failure_recycles_the_session(self):
        adapter = _make_adapter()
        adapter._running = True
        old, fresh = _session(), _session()
        adapter._send_session = old

        async def _fail(*args, **kwargs):
            raise ConnectionError("Cannot connect to host ilinkai.weixin.qq.com:443 ssl:default [None]")

        with patch.object(weixin, "_new_session", return_value=fresh), \
                patch.object(adapter, "_send_file", _fail):
            result = asyncio.run(adapter._send_file_result("chat", "path.bin", "", "send_document"))

        self.assertFalse(result.success)
        self.assertIs(adapter._send_session, fresh)
        old.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
