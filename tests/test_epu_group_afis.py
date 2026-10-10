import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "EPU_Group_AFIS.py"
SPEC = importlib.util.spec_from_file_location("epu_group_afis", MODULE_PATH)
epu_group_afis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(epu_group_afis)


def test_parse_metadata_file_returns_beam_shift(tmp_path):
    metadata_path = tmp_path / "FoilHole_1_Data_2.xml"
    metadata_path.write_text(
        """<root xmlns:fei=\"http://schemas.datacontract.org/2004/07/Fei.SharedObjects\">
        <fei:microscopeData>
            <fei:optics>
                <fei:BeamShift>
                    <fei:Value>0.10</fei:Value>
                    <fei:Value>-0.20</fei:Value>
                </fei:BeamShift>
            </fei:optics>
        </fei:microscopeData>
        </root>""",
        encoding="utf-8",
    )

    namespace = {"fei": "http://schemas.datacontract.org/2004/07/Fei.SharedObjects"}

    assert epu_group_afis._parse_metadata_file(metadata_path, namespace) == [0.1, -0.2]


def test_index_movie_files_matches_metadata_root(tmp_path):
    movie_dir = tmp_path / "movies"
    movie_dir.mkdir()
    movie_path = movie_dir / "FoilHole_1_Data_2_Fractions.mrc"
    movie_path.write_bytes(b"movie")

    lookup = epu_group_afis._index_movie_files(str(movie_dir), "mrc", ["FoilHole_1_Data_2"])

    assert lookup["FoilHole_1_Data_2"] == str(movie_path)


def test_configure_sklearn_threads_sets_environment(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    epu_group_afis._configure_sklearn_threads(4)

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert os.environ[key] == "4"


def test_cli_invokes_processing_with_temp_dataset(tmp_path):
    metadata_dir = tmp_path / "xml"
    metadata_dir.mkdir()
    metadata_path = metadata_dir / "FoilHole_1_Data_1.xml"
    metadata_path.write_text(
        """<root xmlns:fei=\"http://schemas.datacontract.org/2004/07/Fei.SharedObjects\">
        <fei:microscopeData>
            <fei:optics>
                <fei:BeamShift>
                    <fei:Value>0.10</fei:Value>
                    <fei:Value>0.20</fei:Value>
                </fei:BeamShift>
            </fei:optics>
        </fei:microscopeData>
        </root>""",
        encoding="utf-8",
    )
    metadata_path2 = metadata_dir / "FoilHole_2_Data_1.xml"
    metadata_path2.write_text(
        """<root xmlns:fei=\"http://schemas.datacontract.org/2004/07/Fei.SharedObjects\">
        <fei:microscopeData>
            <fei:optics>
                <fei:BeamShift>
                    <fei:Value>-0.10</fei:Value>
                    <fei:Value>-0.20</fei:Value>
                </fei:BeamShift>
            </fei:optics>
        </fei:microscopeData>
        </root>""",
        encoding="utf-8",
    )

    output_path = tmp_path / "movies.star"
    exit_code = epu_group_afis.cli([
        "--xml_dir",
        str(metadata_dir),
        "--n_clusters",
        "2",
        "--quiet",
        "--output_fn",
        str(output_path),
        "--workers",
        "1",
    ])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8").count("# version 50001") == 2
