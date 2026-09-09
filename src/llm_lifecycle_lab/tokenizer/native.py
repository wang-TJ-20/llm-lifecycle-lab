"""Train and load a byte-level BPE tokenizer for native model routes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from llm_lifecycle_lab.contracts import RecordKind, TokenizerManifest
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.data.prepare import (
    load_data_manifest,
    verify_data_manifest,
)
from llm_lifecycle_lab.exceptions import (
    ArtifactError,
    ContractError,
    DataValidationError,
)

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
CHAT_START_TOKEN = "<|im_start|>"
CHAT_END_TOKEN = "<|im_end|>"
RESERVED_CONTROL_TOKENS = tuple(f"<|reserved_{index}|>" for index in range(8))
SPECIAL_TOKENS = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    CHAT_START_TOKEN,
    CHAT_END_TOKEN,
    *RESERVED_CONTROL_TOKENS,
)
NATIVE_CHAT_TEMPLATE_VERSION = "native-chat-v1"
NATIVE_CHAT_TEMPLATE = (
    "{%- for message in messages %}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}"
)
_CHAT_ROLES = frozenset({"system", "user", "assistant"})


class NativeTokenizer:
    """Small wrapper that binds tokenizer.json to its immutable manifest."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        manifest: TokenizerManifest,
    ) -> None:
        self._tokenizer = tokenizer
        self.manifest = manifest

    @classmethod
    def from_directory(cls, path: str | Path) -> NativeTokenizer:
        root = Path(path)
        tokenizer_path = root / "tokenizer.json"
        manifest_path = root / "tokenizer_manifest.json"
        if not tokenizer_path.is_file() or not manifest_path.is_file():
            raise ArtifactError(
                f"tokenizer directory must contain tokenizer.json and "
                f"tokenizer_manifest.json: {root}"
            )
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = TokenizerManifest.from_dict(manifest_value)
        except (OSError, TypeError, ValueError, KeyError, ContractError) as exc:
            raise ArtifactError(
                f"invalid tokenizer manifest {manifest_path}: {exc}"
            ) from exc
        actual_sha256 = sha256_file(tokenizer_path)
        if actual_sha256 != manifest.content_sha256:
            raise ArtifactError(
                f"tokenizer hash mismatch: expected {manifest.content_sha256}, "
                f"got {actual_sha256}"
            )
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise ArtifactError(
                f"cannot load tokenizer {tokenizer_path}: {exc}"
            ) from exc
        if tokenizer.get_vocab_size() != manifest.vocab_size:
            raise ArtifactError("tokenizer vocabulary size does not match its manifest")
        _verify_native_protocol(tokenizer, manifest)
        return cls(tokenizer, manifest)

    @property
    def vocab_size(self) -> int:
        return self.manifest.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._special_token_id(PAD_TOKEN)

    @property
    def bos_token_id(self) -> int:
        return self._special_token_id(BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        return self._special_token_id(EOS_TOKEN)

    @property
    def unk_token_id(self) -> int:
        return self._special_token_id(UNK_TOKEN)

    @property
    def chat_start_token_id(self) -> int:
        return self._special_token_id(CHAT_START_TOKEN)

    @property
    def chat_end_token_id(self) -> int:
        return self._special_token_id(CHAT_END_TOKEN)

    @property
    def chat_template(self) -> str:
        value = self.manifest.chat_template
        if value is None:
            raise ArtifactError("native tokenizer has no chat template")
        return value

    @property
    def chat_template_version(self) -> str:
        value = self.manifest.chat_template_version
        if value is None:
            raise ArtifactError("native tokenizer has no chat template version")
        return value

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.encode(
            render_native_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _special_token_id(self, token: str) -> int:
        return int(self.manifest.special_tokens[token])


def render_native_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render the stable Native chat-v1 wire format."""

    if not messages:
        raise ContractError("native chat requires at least one message")

    rendered: list[str] = []
    expected_role = "user"
    last_role = ""
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in _CHAT_ROLES:
            raise ContractError("native chat role must be system, user, or assistant")
        if not isinstance(content, str) or not content:
            raise ContractError("native chat content must be a non-empty string")
        if any(token in content for token in SPECIAL_TOKENS):
            raise ContractError("native chat content contains a control token")

        if role == "system":
            if index != 0:
                raise ContractError("native chat system message must be first")
        else:
            if role != expected_role:
                raise ContractError(
                    f"native chat expected {expected_role} at message {index}"
                )
            expected_role = "assistant" if role == "user" else "user"

        rendered.append(f"{CHAT_START_TOKEN}{role}\n{content}{CHAT_END_TOKEN}\n")
        last_role = role

    if add_generation_prompt:
        if last_role != "user":
            raise ContractError(
                "native generation prompt requires a final user message"
            )
        rendered.append(f"{CHAT_START_TOKEN}assistant\n")
    return "".join(rendered)


def train_native_tokenizer(
    data_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_id: str,
    vocab_size: int = 16_384,
    min_frequency: int = 2,
) -> TokenizerManifest:
    """Train a byte-level BPE tokenizer from the pretraining train split."""

    minimum_vocab_size = len(pre_tokenizers.ByteLevel.alphabet()) + len(SPECIAL_TOKENS)
    if vocab_size < minimum_vocab_size:
        raise DataValidationError(
            f"vocab_size must be at least {minimum_vocab_size} for "
            "byte-level BPE and the Native protocol"
        )
    if min_frequency <= 0:
        raise DataValidationError("min_frequency must be positive")
    if not tokenizer_id.strip():
        raise DataValidationError("tokenizer_id must not be empty")

    manifest_path = Path(data_manifest_path)
    data_manifest = load_data_manifest(manifest_path)
    if data_manifest.record_kind is not RecordKind.PRETRAIN:
        raise DataValidationError(
            "native tokenizer training requires a pretrain Data Manifest"
        )
    failures = verify_data_manifest(manifest_path)
    if failures:
        raise DataValidationError("; ".join(failures))

    train_split = next(split for split in data_manifest.splits if split.name == "train")
    train_path = manifest_path.parent / train_split.path
    tokenizer = _new_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        _iter_pretrain_text(train_path),
        trainer=trainer,
        length=train_split.records,
    )

    target = Path(output_dir)
    if target.exists():
        raise ArtifactError(
            f"tokenizer output already exists; refusing to overwrite: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )
    try:
        tokenizer_path = temporary / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        content_sha256 = sha256_file(tokenizer_path)
        special_tokens = {
            token: _require_token_id(tokenizer, token) for token in SPECIAL_TOKENS
        }
        manifest = TokenizerManifest(
            tokenizer_id=tokenizer_id,
            revision=content_sha256,
            vocab_size=tokenizer.get_vocab_size(),
            content_sha256=content_sha256,
            source_data_sha256=sha256_file(manifest_path),
            special_tokens=special_tokens,
            trainer_config={
                "algorithm": "byte-level-bpe",
                "normalizer": "NFKC",
                "protocol_version": NATIVE_CHAT_TEMPLATE_VERSION,
                "requested_vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "train_split_sha256": train_split.sha256,
            },
            chat_template=NATIVE_CHAT_TEMPLATE,
            chat_template_version=NATIVE_CHAT_TEMPLATE_VERSION,
        )
        _atomic_json(
            temporary / "tokenizer_manifest.json",
            manifest.to_dict(),
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _new_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _iter_pretrain_text(path: Path) -> Iterator[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise DataValidationError(
                        f"{path} line {line_number} has no usable text"
                    )
                yield text
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid JSON in prepared split {path}") from exc
    except OSError as exc:
        raise DataValidationError(f"cannot read prepared split {path}: {exc}") from exc


def _require_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ArtifactError(f"trained tokenizer is missing special token {token}")
    return token_id


def _verify_native_protocol(
    tokenizer: Tokenizer,
    manifest: TokenizerManifest,
) -> None:
    if set(manifest.special_tokens) != set(SPECIAL_TOKENS):
        raise ArtifactError(
            "tokenizer is incompatible with the Native chat-v1 protocol"
        )
    for token, token_id in manifest.special_tokens.items():
        if tokenizer.token_to_id(token) != token_id:
            raise ArtifactError(f"tokenizer special token mapping mismatch for {token}")
    if (
        manifest.chat_template != NATIVE_CHAT_TEMPLATE
        or manifest.chat_template_version != NATIVE_CHAT_TEMPLATE_VERSION
    ):
        raise ArtifactError(
            "tokenizer chat template is incompatible with "
            f"{NATIVE_CHAT_TEMPLATE_VERSION}"
        )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot write tokenizer manifest: {exc}") from exc
