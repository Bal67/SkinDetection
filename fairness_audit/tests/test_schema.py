import io

import pytest

from fairness_audit.schema import SchemaError, load_predictions

VALID_CSV = """image_id,true_label,predicted_label,confidence,fitzpatrick_scale
img_001,acne,acne,0.91,1
img_002,acne,eczema,0.62,3
img_003,eczema,eczema,0.88,5
img_004,melanoma,melanoma,0.77,6
img_005,melanoma,acne,0.55,
"""


def test_valid_input_happy_path():
    df = load_predictions(io.StringIO(VALID_CSV))
    assert len(df) == 5
    assert list(df["fitzpatrick_scale"]) == [1, 3, 5, 6, -1]
    assert df["fitzpatrick_scale"].dtype.kind == "i"


def test_accepts_file_path(tmp_path):
    path = tmp_path / "preds.csv"
    path.write_text(VALID_CSV)
    df = load_predictions(str(path))
    assert len(df) == 5


def test_missing_required_column():
    csv = "image_id,true_label,confidence,fitzpatrick_scale\nimg_001,acne,0.9,1\n"
    with pytest.raises(SchemaError, match="predicted_label"):
        load_predictions(io.StringIO(csv))


def test_empty_file():
    with pytest.raises(SchemaError):
        load_predictions(io.StringIO(""))


def test_header_only_no_rows():
    csv = "image_id,true_label,predicted_label,fitzpatrick_scale\n"
    with pytest.raises(SchemaError, match="no data rows"):
        load_predictions(io.StringIO(csv))


def test_non_numeric_fitzpatrick_scale():
    csv = (
        "image_id,true_label,predicted_label,fitzpatrick_scale\n"
        "img_001,acne,acne,notanumber\n"
    )
    with pytest.raises(SchemaError, match="non-numeric"):
        load_predictions(io.StringIO(csv))


def test_out_of_range_fitzpatrick_scale():
    csv = (
        "image_id,true_label,predicted_label,fitzpatrick_scale\n"
        "img_001,acne,acne,9\n"
    )
    with pytest.raises(SchemaError, match="out-of-range"):
        load_predictions(io.StringIO(csv))


def test_duplicate_image_id():
    csv = (
        "image_id,true_label,predicted_label,fitzpatrick_scale\n"
        "img_001,acne,acne,1\n"
        "img_001,eczema,eczema,2\n"
    )
    with pytest.raises(SchemaError, match="unique"):
        load_predictions(io.StringIO(csv))


def test_blank_required_field():
    csv = (
        "image_id,true_label,predicted_label,fitzpatrick_scale\n"
        "img_001,,acne,1\n"
    )
    with pytest.raises(SchemaError, match="true_label"):
        load_predictions(io.StringIO(csv))


def test_confidence_out_of_range():
    csv = (
        "image_id,true_label,predicted_label,confidence,fitzpatrick_scale\n"
        "img_001,acne,acne,1.5,1\n"
    )
    with pytest.raises(SchemaError, match="confidence"):
        load_predictions(io.StringIO(csv))


def test_missing_confidence_column_is_optional():
    csv = (
        "image_id,true_label,predicted_label,fitzpatrick_scale\n"
        "img_001,acne,acne,1\n"
    )
    df = load_predictions(io.StringIO(csv))
    assert "confidence" not in df.columns or df["confidence"].isna().all()
