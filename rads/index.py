import math
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import and_

from .config.loader import get_dataroot, load_config
from .config.tree import Config
from .logging import log
from .typing import PathLike, PathLikeOrFile
from .utility import ensure_open

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.base import Connection

# increment this each time the schema is changed so the index will be rebuilt
_SCHEMA_VERSION = 1
_FLOAT_PRECISION = 23
_DOUBLE_PRECISION = 53


metadata = MetaData()

# the schema for this table must never be changed
keystore = Table(
    "keystore",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("key", String(64), nullable=False, unique=True),
    Column("value", String(1024)),
)

index_files = Table(
    "index_files",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("size", Integer, nullable=False),
    Column("modified", Float(_DOUBLE_PRECISION), nullable=False),
    Column(
        "cycle_id", None, ForeignKey("cycles.id", ondelete="cascade"), nullable=False
    ),
)

satellites = Table(
    "satellites",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("satellite", String(2), nullable=False, unique=True),
)

phases = Table(
    "phases",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("phase", String(1), nullable=False),
    Column(
        "satellite_id",
        None,
        ForeignKey("satellites.id", ondelete="cascade"),
        nullable=False,
    ),
    UniqueConstraint("satellite_id", "phase"),
)

cycles = Table(
    "cycles",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("cycle", Integer, nullable=False),
    Column(
        "phase_id", None, ForeignKey("phases.id", ondelete="cascade"), nullable=False
    ),
    UniqueConstraint("phase_id", "cycle"),
)

passes = Table(
    "passes",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("pass", Integer, nullable=False),
    Column("samples", Integer, nullable=False),
    Column("first_measurement_time", Float(_DOUBLE_PRECISION), nullable=False),
    Column("last_measurement_time", Float(_DOUBLE_PRECISION), nullable=False),
    Column("equator_crossing_time", Float(_DOUBLE_PRECISION), nullable=False),
    Column("equator_crossing_longitude", Float(_DOUBLE_PRECISION), nullable=False),
    Column(
        "cycle_id", None, ForeignKey("cycles.id", ondelete="cascade"), nullable=False
    ),
    UniqueConstraint("cycle_id", "pass"),
)


class Index:
    def __init__(
        self, dataroot: Optional[PathLike] = None, url: str = "sqlite:///:memory:"
    ):
        # if using sqlite ensure the containing directory exists
        if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
            Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
        self._dataroot = get_dataroot(dataroot)
        self._engine = create_engine(url)
        version = self.version
        if not version or version < _SCHEMA_VERSION:
            self.clear()

    @property
    def engine(self) -> "Engine":
        return self._engine

    @property
    def version(self) -> Optional[int]:
        with self._engine.begin() as connection:
            try:
                return int(
                    connection.execute(
                        select([keystore.c.value]).where(
                            keystore.c.key == "SCHEMA_VERSION"
                        )
                    ).fetchone()[0]
                )
            except (TypeError, OperationalError):
                return None

    def __bool__(self):
        return len(self) != 0 and self.version == _SCHEMA_VERSION

    def __len__(self) -> int:
        with self._engine.begin() as connection:
            return connection.execute(
                select([func.count()]).select_from(passes)
            ).first()[0]

    def clear(self) -> None:
        # clear the database
        table_names = self._engine.table_names()
        with self._engine.begin() as connection:
            for table in table_names:
                connection.execute(f"DROP TABLE IF EXISTS {table};")
        # create tables
        metadata.create_all(self._engine)
        # set schema version
        with self._engine.begin() as connection:
            stmt = keystore.insert().values(
                key="SCHEMA_VERSION", value=str(_SCHEMA_VERSION)
            )
            connection.execute(stmt)

    def update(self) -> int:
        log.info(f"Updating cache {self._engine.url}")
        with self._engine.begin() as connection:
            data = self._pack_for_database(self._new_passes(connection))
            num_updated = len(data)
            if data:
                log.info(f"{num_updated} new/modified passes")
                connection.execute(passes.insert(), data)
            return num_updated

    @staticmethod
    def _pack_for_database(
        tuples: Iterator[Tuple[int, int, int, float, float, float, float]]
    ) -> Sequence[Mapping[str, Union[int, float]]]:
        data: List = []
        for cycle_id, pass_, samples, first, last, ctime, ctime in tuples:
            data.append(
                {
                    "cycle_id": cycle_id,
                    "pass": pass_,
                    "samples": samples,
                    "first_measurement_time": first,
                    "last_measurement_time": last,
                    "equator_crossing_time": ctime,
                    "equator_crossing_longitude": ctime,
                }
            )
        return data

    def _new_passes(
        self, connection: "Connection"
    ) -> Iterator[Tuple[int, int, int, float, float, float, float]]:
        for satellite in self._satellites():
            satellite_id = self._insert_satellite(connection, satellite)
            for phase in self._phases(satellite):
                phase_id = self._insert_phase(connection, satellite_id, phase)
                for cycle in self._cycles(satellite, phase):
                    cycle_id = self._insert_cycle(connection, phase_id, cycle)
                    path = self._passindex_path(satellite, phase, cycle)
                    if path.is_file():
                        yield from self._new_from_passindex(connection, cycle_id, path)

    def _passindex_path(self, satellite: str, phase: str, cycle: int) -> Path:
        return self._dataroot / satellite / phase / f"c{cycle:03d}" / ".passindex"

    def _new_from_passindex(self, connection: "Connection", cycle_id: int, path: Path):
        # get stats of passindex file
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime

        # compare with old stats
        old_size, old_mtime = self._old_passindex_stats(connection, cycle_id)
        if size == old_size and math.isclose(mtime, old_mtime, rel_tol=0, abs_tol=1):
            return  # file is already indexed

        # remove data from old passindex (this is faster than upsert)
        if old_size != 0:
            connection.execute(passes.delete().where(passes.c.cycle_id == cycle_id))
            log.info(f"Cycle updated {path.parent}")
        else:
            log.info(f"New cycle {path.parent}")

        # update/insert passindex file stats
        self._upsert_index_file(connection, cycle_id, size, mtime)

        # yield data from the passindex file
        for line in read_passindex(path):
            yield (cycle_id, *line)

    def _old_passindex_stats(
        self, connection: "Connection", cycle_id: int
    ) -> Tuple[int, float]:
        try:
            old_size, old_mtime = connection.execute(
                select([index_files.c.size, index_files.c.modified]).where(
                    index_files.c.cycle_id == cycle_id
                )
            ).first()
            return old_size, old_mtime
        except TypeError:
            return 0, 0.0

    def _satellites(self) -> Iterator[str]:
        for path in self._dataroot.iterdir():
            satellite = path.name
            if path.is_dir() and len(satellite) == 2 and satellite.isalnum():
                yield satellite

    def _phases(self, satellite: str) -> Iterator[str]:
        for path in (self._dataroot / satellite).iterdir():
            phase = path.name
            # phase must:
            #   * be a directory
            #   * be a single alphanumeric character
            #   * not be a symlink to another phase
            if (
                path.is_dir()
                and len(phase) == 1
                and phase.isalnum()
                and (
                    not path.is_symlink()
                    or path.resolve().parent != path.parent.resolve()
                )
            ):
                yield phase

    def _cycles(self, satellite: str, phase: str) -> Iterator[int]:
        for path in (self._dataroot / satellite / phase).iterdir():
            cycle = path.name
            if path.is_dir and re.match(r"^c\d\d\d$", cycle):
                yield int(cycle[1:])

    @staticmethod
    def _insert_satellite(connection: "Connection", satellite: str) -> int:
        try:
            return connection.execute(
                select([satellites.c.id]).where(satellites.c.satellite == satellite)
            ).first()[0]
        except TypeError:
            return connection.execute(
                satellites.insert(), satellite=satellite
            ).inserted_primary_key[0]

    @staticmethod
    def _insert_phase(connection: "Connection", satellite_id: int, phase: str) -> int:
        try:
            return connection.execute(
                select([phases.c.id]).where(
                    and_(phases.c.satellite_id == satellite_id, phases.c.phase == phase)
                )
            ).first()[0]
        except TypeError:
            return connection.execute(
                phases.insert(), phase=phase, satellite_id=satellite_id
            ).inserted_primary_key[0]

    @staticmethod
    def _insert_cycle(connection: "Connection", phase_id: int, cycle: int) -> int:
        try:
            return connection.execute(
                select([cycles.c.id]).where(
                    and_(cycles.c.phase_id == phase_id, cycles.c.cycle == cycle)
                )
            ).first()[0]
        except TypeError:
            return connection.execute(
                cycles.insert(), cycle=cycle, phase_id=phase_id
            ).inserted_primary_key[0]

    @staticmethod
    def _upsert_index_file(
        connection: "Connection", cycle_id: int, size: int, modified: float
    ) -> None:
        # UPSERT would be better but that is only supported with SQLite and PostgreSQL
        result = connection.execute(
            index_files.update()
            .where(index_files.c.cycle_id == cycle_id)
            .values(size=size, modified=modified)
        )
        if result.rowcount != 1:
            connection.execute(
                index_files.insert(), size=size, modified=modified, cycle_id=cycle_id
            )


def read_passindex(
    file: PathLikeOrFile
) -> Iterator[Tuple[int, int, float, float, float]]:
    with ensure_open(file) as file_:
        for line in file_:
            values = line.split()[2:]
            yield (
                int(values[0]),  # pass
                int(values[5]),  # samples
                float(values[1]),  # first measurement time
                float(values[2]),  # last measurement time
                float(values[3]),  # equator crossing time
                float(values[4]),  # equator crossing longitude
            )


def update_index(config: Optional[Config] = None) -> int:
    if config is None:
        config = load_config()
    index = Index(dataroot=config.dataroot, url=config.cachedb)
    return index.update()
