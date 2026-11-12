"""OpenAI-compatible client that rotates over several API keys.

- Per-minute limit on a key  -> move to the next key (wait only if every key is cooling down).
- Daily limit on a key       -> mark it exhausted for this session and move on.
- All keys exhausted         -> raise AllKeysExhausted; the runner saves and stops cleanly.
"""
import re
import time
from types import SimpleNamespace

import openai


class AllKeysExhausted(RuntimeError):
    pass


MALFORMED = ("tool_use_failed", "output_parse_failed")


def is_malformed(e) -> bool:
    """Server rejected the model's own output (bad tool name, bad arguments, unparsable)."""
    return isinstance(e, openai.BadRequestError) and getattr(e, "code", None) in MALFORMED


DAILY = re.compile(r"per day|\(TPD\)|\(RPD\)|\(ASD\)", re.I)
WAIT = re.compile(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)s", re.I)


def _wait_seconds(msg: str) -> float:
    m = WAIT.search(msg)
    if not m:
        return 10.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


class RotatingClient:
    def __init__(self, keys, base_url, timeout=120, log=print):
        keys = [k.strip() for k in keys if k and k.strip()]
        if not keys:
            raise ValueError("No API keys given")
        self.clients = [openai.OpenAI(api_key=k, base_url=base_url, max_retries=0, timeout=timeout) for k in keys]
        self.tags = [f"key#{i + 1}(...{k[-4:]})" for i, k in enumerate(keys)]
        self.exhausted = set()
        self.cool_until = [0.0] * len(keys)
        self.i = 0
        self.log = log
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def status(self):
        return f"{len(self.clients) - len(self.exhausted)}/{len(self.clients)} keys still usable"

    def _next_key(self):
        n = len(self.clients)
        for step in range(1, n + 1):
            j = (self.i + step) % n
            if j not in self.exhausted:
                self.i = j
                return
        raise AllKeysExhausted("All API keys have reached their daily limit.")

    def _create(self, **kwargs):
        malformed = 0
        server_errors = 0
        while True:
            if len(self.exhausted) == len(self.clients):
                raise AllKeysExhausted("All API keys have reached their daily limit.")
            if self.i in self.exhausted:
                self._next_key()
            now = time.time()
            if self.cool_until[self.i] > now:
                live = [j for j in range(len(self.clients)) if j not in self.exhausted]
                ready = [j for j in live if self.cool_until[j] <= now]
                if ready:
                    self.i = ready[0]
                else:
                    time.sleep(min(self.cool_until[j] for j in live) - now + 0.5)
                    continue
            try:
                return self.clients[self.i].chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                msg = str(e)
                if DAILY.search(msg):
                    self.exhausted.add(self.i)
                    self.log(f"  {self.tags[self.i]} hit its daily limit -> {self.status()}")
                    if len(self.exhausted) == len(self.clients):
                        raise AllKeysExhausted("All API keys have reached their daily limit.") from e
                    self._next_key()
                else:
                    self.cool_until[self.i] = time.time() + _wait_seconds(msg)
                    self._next_key()
            except openai.BadRequestError as e:
                # gpt-oss on Groq sometimes emits a malformed tool call; resample up to 3 times
                if is_malformed(e) and malformed < 3:
                    malformed += 1
                    continue
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                server_errors += 1
                if server_errors > 6:
                    raise
                time.sleep(min(60, 2 ** server_errors))
