"""Server response boundaries and fatal failures, without model inference."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm.backbone import build_llm
from llm.errors import InferenceContractError


def response(content='{"answer":323}', *, finish="stop", **fields):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish,
                                 message=SimpleNamespace(content=content, **fields))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        system_fingerprint="test-parser")


class TestReasoningContract(unittest.TestCase):
    def llm(self, mode, directory=None):
        with patch.dict(os.environ, {}, clear=True):
            llm = build_llm("qwen", "Qwen/Qwen3-4B", seed=1103,
                            thinking_mode=mode, log_dir=directory)
        llm._client = Mock()
        return llm

    def test_both_server_field_names_keep_reasoning_out_of_content_and_logs(self):
        for field in ("reasoning", "reasoning_content"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                llm = self.llm("on", tmp)
                llm._client.chat.completions.create.return_value = response(**{field: "private reasoning fixture"})
                proof = llm.validate_server()
                self.assertTrue(proof["server_separation_verified"])
                self.assertEqual(proof["reasoning_fields"], [field])
                request = llm._client.chat.completions.create.call_args.kwargs
                self.assertIs(request["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
                logged = (Path(tmp) / "llm_calls.jsonl").read_text()
                self.assertNotIn("private reasoning fixture", logged)
                self.assertEqual(json.loads(logged)["response_metadata"]["thinking_mode"], "on")
                self.assertEqual(llm.usage_summary()["total_tokens"], 30)

    def test_non_thinking_requires_empty_reasoning_and_passes_false(self):
        llm = self.llm("off")
        llm._client.chat.completions.create.return_value = response(reasoning_content="", reasoning=None)
        self.assertEqual(llm.validate_server()["status"], "passed")
        request = llm._client.chat.completions.create.call_args.kwargs
        self.assertIs(request["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)

    def test_real_sdk_preserves_server_reasoning_field(self):
        import httpx
        from openai import OpenAI
        for field in ("reasoning", "reasoning_content"):
            with self.subTest(field=field):
                requests = []
                def handler(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        "id": "test", "object": "chat.completion", "created": 0,
                        "model": "Qwen/Qwen3-4B", "choices": [{"index": 0,
                        "finish_reason": "stop", "message": {"role": "assistant",
                        "content": '{"answer":323}', field: "separate reasoning fixture"}}]})
                llm = self.llm("on")
                with OpenAI(api_key="local-no-key", base_url="http://test.invalid/v1",
                            http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                    llm._client = client
                    self.assertEqual(llm.validate_server()["reasoning_fields"], [field])
                self.assertIs(requests[0]["chat_template_kwargs"]["enable_thinking"], True)
                self.assertEqual(requests[0]["seed"], 1103)

    def test_malformed_or_truncated_outputs_fail_closed(self):
        missing_message = response()
        missing_message.choices[0].message = None
        no_choices = response()
        no_choices.choices = None
        cases = [
            ("off", no_choices), ("off", missing_message),
            ("on", response()),
            ("off", response(reasoning="unexpected")),
            ("on", response('<think>raw</think>{"answer":323}', reasoning="split")),
            ("on", response('raw</think>{"answer":323}', reasoning="split")),
            ("off", response("<THINK>unfinished")),
            ("on", response(None, reasoning="still thinking")),
            ("on", response("", reasoning="still thinking")),
            ("on", response(finish="length", reasoning="split")),
            ("off", response(finish="content_filter")),
            ("off", response(reasoning=["wrong field type"])),
        ]
        for mode, result in cases:
            with self.subTest(mode=mode, result=result), tempfile.TemporaryDirectory() as tmp:
                llm = self.llm(mode, tmp)
                llm._client.chat.completions.create.return_value = result
                with self.assertRaises(InferenceContractError):
                    llm.call_text(system_prompt="JSON", payload={})
                self.assertEqual(llm.failed_call_count, 1)
                self.assertEqual(llm.usage_summary()["total_tokens"], 30)
                self.assertIn("failed_call", (Path(tmp) / "llm_calls.jsonl").read_text())

    def test_preflight_rejects_non_json_final_answer(self):
        llm = self.llm("off")
        llm._client.chat.completions.create.return_value = response("The answer is 323.")
        with self.assertRaisesRegex(InferenceContractError, "Preflight"):
            llm.validate_server()

    def test_configuration_requires_explicit_boolean_qwen3_mode(self):
        for env in ({}, {"OPEN_SOURCE_THINKING": "true"},
                    {"OPEN_SOURCE_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":"false"}}'},
                    {"OPEN_SOURCE_THINKING": "on", "OPEN_SOURCE_EXTRA_BODY":
                     '{"chat_template_kwargs":{"enable_thinking":false}}'}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit):
                    build_llm("qwen", "Qwen/Qwen3-4B")
        with patch.dict(os.environ, {"OPEN_SOURCE_THINKING": "off"}, clear=True):
            self.assertEqual(build_llm("qwen", "Qwen/Qwen3-32B", thinking_mode="on").thinking_mode, "on")
            with self.assertRaises(SystemExit):
                build_llm("llama", "meta-llama/Llama-3.1-8B-Instruct", thinking_mode="off")

    def test_contract_failure_is_not_converted_into_policy_fallback(self):
        from test_redesign import Test43OutputSafety
        from llm.policy_agent import DirectionMemory
        from llm.policy_agent import Policy
        fixture = Test43OutputSafety()
        agent = fixture._agent(InferenceContractError("bad separation"))
        calls = [lambda: agent.discover_direction(candidates=[fixture._cand()]),
                 lambda: agent.discover_daily_direction(candidate=fixture._cand()),
                 lambda: agent.reflect_direction(DirectionMemory()),
                 lambda: agent._verify_critical_news(("A", "B"), {}, "bankruptcy"),
                 lambda: agent.detect_relationship_break(
                     pair=("A", "B"), recent_corr=0.9, signal_z=0.5, adf_pvalue=0.01,
                     coint_pvalue=0.01, half_life=10., news_summary={}, pair_memory=None,
                     policy=Policy())]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(InferenceContractError):
                call()


if __name__ == "__main__":
    unittest.main()
