from __future__ import annotations

import sys
import types
from collections.abc import Iterable

import pytest

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role
from mirrorbench.core.models.registry import (
    DatasetInfo,
    JudgeInfo,
    MetricInfo,
    ModelClientInfo,
    UserProxyAdapterInfo,
)
from mirrorbench.core.models.run import MetricValue
from mirrorbench.core.registry import (
    GROUPS,
    BaseDatasetLoader,
    BaseJudge,
    BaseMetric,
    BaseModelClient,
    BaseUserProxyAdapter,
    filter_by_capability,
    filter_by_task,
    filter_metrics_requiring_judge,
    register_dataset,
    register_judge,
    register_metric,
    register_model_client,
    register_user_proxy,
    registry,
)
from mirrorbench.core.registry.entries import RegistryEntry


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterable[None]:
    original_entries = {group: list(registry.list_entries(group)) for group in GROUPS}
    registry.clear()
    yield
    registry.clear()
    for _group, entries in original_entries.items():
        for entry in entries:
            registry.register(entry)


class DummyAdapter(BaseUserProxyAdapter):
    info = UserProxyAdapterInfo(name="dummy:echo", capabilities={"chat"})

    def spawn(self, *, config, run_id):  # type: ignore[override]
        return {"config": config.name, "run_id": run_id}


class DummyDataset(BaseDatasetLoader):
    info = DatasetInfo(name="dummy:set", supported_tasks={"qa"}, splits={"test"})

    def episodes(self, *, spec, split, limit=None):  # type: ignore[override]
        del spec, split, limit
        message = Message(role=Role.USER, content="hello")
        spec = EpisodeSpec(episode_id="ep-1", task_tag="qa", chat_history=[message])
        yield spec


class DummyMetric(BaseMetric):
    info = MetricInfo(name="dummy:score", supported_tasks={"qa"})

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:  # type: ignore[override]
        del episode
        return MetricValue(metric_name=self.info.name, values=[0.4, 0.6])


class DummyJudge(BaseJudge):
    info = JudgeInfo(name="dummy:judge")

    def score(self, episode: EpisodeArtifact):  # type: ignore[override]
        del episode
        return JudgeVerdict(score=1.0, label="pass")


class DummyModelClient(BaseModelClient):
    info = ModelClientInfo(
        name="client:dummy/chat",
        provider="dummy",
        capabilities={"chat"},
        models={"dummy-model"},
    )

    def __call__(self) -> str:  # pragma: no cover - used in tests
        return "client-instance"


@pytest.fixture
def registered_components() -> None:
    register_user_proxy(
        name="dummy:echo",
        metadata=DummyAdapter.info,
    )(DummyAdapter)
    register_dataset(
        name="dummy:set",
        metadata=DummyDataset.info,
    )(DummyDataset)
    register_metric(
        name="dummy:score",
        metadata=DummyMetric.info,
    )(DummyMetric)
    register_judge(
        name="dummy:judge",
        metadata=DummyJudge.info,
    )(DummyJudge)
    register_model_client(
        name="client:dummy/chat",
        metadata=DummyModelClient.info,
    )(DummyModelClient)


def test_decorator_registration(registered_components: None) -> None:
    entry = registry.get("user_proxies", "dummy:echo")
    assert isinstance(entry, RegistryEntry)
    assert isinstance(entry.metadata, UserProxyAdapterInfo)
    assert entry.metadata.capabilities == {"chat"}

    factory = registry.factory("user_proxies", "dummy:echo")
    adapter = factory()
    assert isinstance(adapter, DummyAdapter)

    client_entry = registry.get("model_clients", "client:dummy/chat")
    assert isinstance(client_entry.metadata, ModelClientInfo)
    client_factory = registry.factory("model_clients", "client:dummy/chat")
    client_instance = client_factory()
    assert isinstance(client_instance, DummyModelClient)


def test_duplicate_registration_raises(registered_components: None) -> None:
    with pytest.raises(RegistryError):
        registry.register_factory(
            "user_proxies",
            "dummy:echo",
            DummyAdapter,
        )
    with pytest.raises(RegistryError):
        registry.register_factory(
            "model_clients",
            "client:dummy/chat",
            DummyModelClient,
        )


def test_register_lazy_factory() -> None:
    module_name = "tests.core.registry.lazy_plugin"
    module = types.ModuleType(module_name)

    def lazy_factory() -> str:
        return "lazy"

    module.lazy_factory = lazy_factory
    sys.modules[module_name] = module
    registry.register_lazy(
        "metrics",
        "lazy:metric",
        f"{module_name}:lazy_factory",
        metadata=MetricInfo(name="lazy:metric"),
    )
    try:
        factory = registry.factory("metrics", "lazy:metric")
        assert factory() == "lazy"
    finally:
        sys.modules.pop(module_name, None)


def test_filters(registered_components: None) -> None:
    tasks = filter_by_task(registry.list_entries("datasets"), "qa")
    assert tasks and tasks[0].name == "dummy:set"

    metrics = filter_metrics_requiring_judge(registry.list_entries("metrics"))
    assert metrics == []

    proxies = filter_by_capability(registry.list_entries("user_proxies"), "chat")
    assert proxies and proxies[0].name == "dummy:echo"


def test_snapshot_returns_serializable_dict(registered_components: None) -> None:
    snap = registry.snapshot()
    assert "user_proxies" in snap
    assert any(entry["name"] == "dummy:echo" for entry in snap["user_proxies"])
    assert any(entry["name"] == "client:dummy/chat" for entry in snap["model_clients"])


def test_registry_entry_round_trip(registered_components: None) -> None:
    entry = registry.get("user_proxies", "dummy:echo")
    payload = entry.to_dict(include_factory=True)
    recreated = RegistryEntry.from_dict(payload)
    assert recreated.group == entry.group
    assert recreated.name == entry.name
    assert recreated.lazy_import == payload.get("lazy_import")
    assert recreated.resolve_factory() is entry.resolve_factory()


def test_base_metric_default_aggregate() -> None:
    metric = DummyMetric()
    value = metric.evaluate(
        EpisodeArtifact(
            spec=EpisodeSpec(
                episode_id="agg",
                task_tag="qa",
                chat_history=[Message(role=Role.USER, content="hi")],
            )
        )
    )
    aggregate = metric.aggregate([value])
    assert aggregate.mean == pytest.approx(0.5)
    assert aggregate.sample_size == len(value.values)


def test_load_namespace_registers_entries() -> None:
    module_name = "tests.core.registry.plugin_module"
    module = types.ModuleType(module_name)
    code = """
from mirrorbench.core.registry import register_metric, BaseMetric
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.run import MetricValue

@register_metric(name="plugin:metric", metadata=MetricInfo(name="plugin:metric"))
class PluginMetric(BaseMetric):
    info = MetricInfo(name="plugin:metric")

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:  # type: ignore[override]
        del episode
        return MetricValue(metric_name=self.info.name, values=[1.0])
"""
    exec(code, module.__dict__)
    sys.modules[module_name] = module
    try:
        registry.load_namespace(module_name)
        entry = registry.get("metrics", "plugin:metric")
        assert isinstance(entry.metadata, MetricInfo)
    finally:
        sys.modules.pop(module_name, None)


def test_load_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEntryPoint:
        def __init__(self, name: str, hook):
            self.name = name
            self._hook = hook

        def load(self):
            return self._hook

    class DummyCollection:
        def __init__(self, entries):
            self._entries = entries

        def select(self, group: str):
            return self._entries

    def hook(catalog):
        catalog.register_factory(
            "judges",
            "entrypoint:judge",
            DummyJudge,
            metadata=JudgeInfo(name="entrypoint:judge"),
        )

    monkeypatch.setattr(
        "mirrorbench.core.registry.loader.entry_points",
        lambda: DummyCollection([DummyEntryPoint("judge", hook)]),
    )

    registry.load_entrypoints("mirrorbench.plugins")
    entry = registry.get("judges", "entrypoint:judge")
    assert isinstance(entry.metadata, JudgeInfo)
