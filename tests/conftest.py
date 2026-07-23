from __future__ import annotations
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _ensure_stubs() -> None:
    try:
        import langchain_ollama  
    except ImportError:
        m = _stub_module("langchain_ollama")

        class _StubChatOllama:
            def __init__(self, *args, **kwargs):
                pass

            def bind_tools(self, *_args, **_kwargs):
                return self

            def invoke(self, *_args, **_kwargs):
                raise RuntimeError("stub ChatOllama: no real Ollama in this environment")

        m.ChatOllama = _StubChatOllama

        try:
            import langchain_groq  
        except ImportError:
            m = _stub_module("langchain_groq")

            class _StubChatGroq:
                def __init__(self, *args, **kwargs):
                    pass

                def bind_tools(self, *_args, **_kwargs):
                    return self

                def invoke(self, *_args, **_kwargs):
                    raise RuntimeError("stub ChatGroq: no real Groq credentials in this environment")

            m.ChatGroq = _StubChatGroq

    try:
        import langchain_huggingface 
    except ImportError:
        m = _stub_module("langchain_huggingface")

        class _StubEmbeddings:
            def __init__(self, *args, **kwargs):
                pass

        m.HuggingFaceEmbeddings = _StubEmbeddings

    try:
        import langchain_chroma 
    except ImportError:
        m = _stub_module("langchain_chroma")

        class _StubChroma:
            def __init__(self, *args, **kwargs):
                pass

            def similarity_search(self, *_args, **_kwargs):
                return []

        m.Chroma = _StubChroma

    try:
        import yfinance  
    except ImportError:
        m = _stub_module("yfinance")

        class _EmptyFrame:
            empty = True

            def dropna(self, *_args, **_kwargs):
                return self

        class _StubTicker:
            def __init__(self, *_args, **_kwargs):
                pass

            def history(self, *_args, **_kwargs):
                return _EmptyFrame()

        m.Ticker = _StubTicker

    try:
        import langchain_text_splitters  
    except ImportError:
        m = _stub_module("langchain_text_splitters")

        class _StubSplitter:
            def __init__(self, *args, **kwargs):
                pass

            def split_documents(self, documents):
                return documents

        m.RecursiveCharacterTextSplitter = _StubSplitter

    try:
        import langchain_core.documents  
    except ImportError:
        core_pkg = sys.modules.get("langchain_core") or _stub_module("langchain_core")
        docs_mod = types.ModuleType("langchain_core.documents")

        class _StubDocument:
            def __init__(self, page_content: str = "", metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

        docs_mod.Document = _StubDocument
        sys.modules["langchain_core.documents"] = docs_mod
        core_pkg.documents = docs_mod  # type: ignore[attr-defined]

    try:
        import langchain_core.messages  
    except ImportError:
        core_pkg = sys.modules.get("langchain_core") or _stub_module("langchain_core")
        msgs_mod = types.ModuleType("langchain_core.messages")

        class _StubMessage:
            def __init__(self, content: str = "", **kwargs):
                self.content = content
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _StubAIMessage(_StubMessage):
            tool_calls: list = []

        class _StubHumanMessage(_StubMessage):
            pass

        class _StubSystemMessage(_StubMessage):
            pass

        class _StubToolMessage(_StubMessage):
            pass

        msgs_mod.AIMessage = _StubAIMessage
        msgs_mod.HumanMessage = _StubHumanMessage
        msgs_mod.SystemMessage = _StubSystemMessage
        msgs_mod.ToolMessage = _StubToolMessage
        sys.modules["langchain_core.messages"] = msgs_mod
        core_pkg.messages = msgs_mod  # type: ignore[attr-defined]

    try:
        import langchain_core.tools 
    except ImportError:
        core_pkg = sys.modules.get("langchain_core") or _stub_module("langchain_core")
        tools_mod = types.ModuleType("langchain_core.tools")

        def _stub_tool_decorator(fn):
            fn.name = fn.__name__
            fn.invoke = lambda args, _fn=fn: _fn(**args)
            return fn

        tools_mod.tool = _stub_tool_decorator
        sys.modules["langchain_core.tools"] = tools_mod
        core_pkg.tools = tools_mod  # type: ignore[attr-defined]

    try:
        import bs4 
    except ImportError:
        pass  # ingest-splitter HTML tests will fail at collection if bs4 is truly absent


_ensure_stubs()
