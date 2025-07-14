import sys, types, builtins, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _stub_module(name, attrs=None):
    module = types.ModuleType(name)
    attrs = attrs or {}
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[name] = module
    return module

# -------------------- Structlog stub --------------------
if 'structlog' not in sys.modules:
    logger_stub = types.ModuleType('structlog_logger')
    for _lvl in ('info', 'debug', 'warning', 'error'):
        setattr(logger_stub, _lvl, lambda *a, **k: None)

    stdlib_stub = _stub_module('structlog.stdlib', {
        'LoggerFactory': object,
        'BoundLogger': object,
        'filter_by_level': lambda *a, **k: None,
        'add_logger_name': lambda *a, **k: None,
        'add_log_level': lambda *a, **k: None,
        'PositionalArgumentsFormatter': lambda *a, **k: None,
    })
    processors_stub = _stub_module('structlog.processors', {
        'TimeStamper': lambda *a, **k: None,
        'StackInfoRenderer': lambda *a, **k: None,
        'format_exc_info': lambda *a, **k: None,
        'UnicodeDecoder': lambda *a, **k: None,
        'JSONRenderer': lambda *a, **k: None,
    })

    structlog_stub = _stub_module('structlog', {
        'get_logger': lambda *a, **k: logger_stub,
        'configure': lambda *a, **k: None,
        'stdlib': stdlib_stub,
        'processors': processors_stub,
    })

# -------------------- Optional heavy libraries --------------------
for heavy_mod in (
    'torch',
    'pyspark',
    'pyspark.sql',
    'bs4',
    'chromadb',
    'sentence_transformers',
    'duckdb',
    'faiss',
    'ollama',
    'prometheus_client',
    'boto3',
    'airflow',
    'airflow.utils',
    'airflow.utils.state',
    'airflow.utils.session',
):
    if heavy_mod not in sys.modules:
        _stub_module(heavy_mod)

# Minimal bs4.BeautifulSoup implementation for scraping tests
if 'bs4' in sys.modules:
    bs4 = sys.modules['bs4']
    class _FakeSoup:
        def __init__(self, *a, **k):
            pass
        def select(self, selector):
            return []
    setattr(bs4, 'BeautifulSoup', _FakeSoup)

# Minimal Airflow DAG stubs to satisfy DAG tests without real Airflow install
if 'airflow' in sys.modules:
    airflow = sys.modules['airflow']
    _stub_module('airflow.models', {'DagBag': object, 'TaskInstance': object})
    _stub_module('airflow.operators')
    airflow.DAG = object
    # airflow.utils.state stub
    _stub_module('airflow.utils.state', {
        'State': type('State', (), {'SUCCESS': 'success', 'FAILED': 'failed'})
    })
    _stub_module('airflow.utils.session')

# Ensure chromadb errors attribute for stubs
if 'chromadb' in sys.modules:
    _stub_module('chromadb.errors', {'DuplicateIDError': Exception})

# sentence_transformers SentenceTransformer stub
if 'sentence_transformers' in sys.modules:
    st = sys.modules['sentence_transformers']
    if not hasattr(st, 'SentenceTransformer'):
        class _FakeST:
            def __init__(self, *a, **k):
                pass
            def encode(self, texts, *a, **k):
                import numpy as np
                return np.zeros((len(texts), 3))
        st.SentenceTransformer = _FakeST

# pyspark.sql types stubs (SparkSession etc) if not real pyspark
if 'pyspark.sql' in sys.modules:
    ps_sql = sys.modules['pyspark.sql']
    if not hasattr(ps_sql, 'SparkSession'):
        class _FakeBuilder:
            def appName(self, *a, **k):
                return self
            def config(self, *a, **k):
                return self
            def getOrCreate(self):
                # Return a minimal object with stop(), read, and sql attributes
                return types.SimpleNamespace(
                    stop=lambda: None,
                    read=types.SimpleNamespace(csv=lambda *a, **k: types.SimpleNamespace(dropna=lambda: None)),
                    sql=lambda q: types.SimpleNamespace()
                )

        class _FakeSparkSession:
            builder = _FakeBuilder()

        ps_sql.SparkSession = _FakeSparkSession 

if 'prometheus_client' in sys.modules:
    pc = sys.modules['prometheus_client']
    pc.generate_latest = lambda *a, **k: b''
    pc.CONTENT_TYPE_LATEST = 'text/plain'
    class _Metric:
        def __init__(self, *a, **k):
            pass
        def inc(self, *a, **k):
            pass
        def observe(self, *a, **k):
            pass
        def set(self, *a, **k):
            pass
        def labels(self, *a, **k):
            return self
    for name in ('Counter', 'Histogram', 'Gauge', 'Info', 'CollectorRegistry', 'REGISTRY'):
        setattr(pc, name, _Metric)

# airflow.utils.dates.days_ago stub
_stub_module('airflow.utils.dates', {'days_ago': lambda n: None})

# airflow.utils.state extended enums
if 'airflow.utils.state' in sys.modules:
    aus = sys.modules['airflow.utils.state']
    aus.DagRunState = type('DagRunState', (), {'SUCCESS': 'success', 'FAILED': 'failed'})
    aus.TaskInstanceState = type('TaskInstanceState', (), {'SUCCESS': 'success', 'FAILED': 'failed'})

# chromadb.config.Settings stub
if 'chromadb' in sys.modules:
    _stub_module('chromadb.config', {'Settings': type('Settings', (), {})})

# pyspark.sql.DataFrame stub
_stub_module('pyspark.sql.dataframe', {'DataFrame': type('DataFrame', (), {})})
if 'pyspark.sql' in sys.modules:
    ps_sql = sys.modules['pyspark.sql']
    if not hasattr(ps_sql, 'DataFrame'):
        ps_sql.DataFrame = type('DataFrame', (), {})

# minio.Minio stub
_stub_module('minio', {'Minio': type('Minio', (), {})})
_stub_module('minio.error', {'S3Error': Exception})

# openlineage client stub
_stub_module('openlineage', {})
_stub_module('openlineage.client', {'OpenLineageClient': type('OpenLineageClient', (), {})})
_stub_module('openlineage.client.event_v2', {'RunEvent': object, 'Dataset': object})
_stub_module('openlineage.client.run', {'RunState': object})

# Ensure embeddings alias
try:
    import importlib
    real_embed = importlib.import_module('src.embeddings.embedder')
    sys.modules['embeddings.embedder'] = real_embed
except Exception:
    pass

# ------------------------------------------------------------------
# Make the project DAG importable under airflow.dags.* for legacy tests
# ------------------------------------------------------------------

import importlib.util, pathlib, os

airflow_dag_path = pathlib.Path(__file__).parent.parent / 'airflow' / 'dags' / 'scrape_etl_embed_dag.py'
if airflow_dag_path.exists():
    spec = importlib.util.spec_from_file_location('airflow.dags.scrape_etl_embed_dag', airflow_dag_path)
    dag_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dag_module)  # type: ignore
    except Exception:
        # If real DAG fails to import due to deps, fall back to empty stub
        dag_module = types.ModuleType('airflow.dags.scrape_etl_embed_dag')
    sys.modules['airflow.dags.scrape_etl_embed_dag'] = dag_module
else:
    _stub_module('airflow.dags.scrape_etl_embed_dag')

# pyspark.sql.types stub
_stub_module('pyspark.sql.types', {'StructType': object, 'StructField': object, 'StringType': object, 'TimestampType': object})

# Alias top-level package names used by fallback imports
if 'src.embeddings' in sys.modules and 'embeddings' not in sys.modules:
    sys.modules['embeddings'] = sys.modules['src.embeddings']
if 'src.common' in sys.modules and 'common' not in sys.modules:
    sys.modules['common'] = sys.modules['src.common']

# torch cuda stub
if 'torch' in sys.modules:
    torch = sys.modules['torch']
else:
    torch = _stub_module('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: False)

# aiohttp stub
_stub_module('aiohttp', {'ClientSession': object}) 