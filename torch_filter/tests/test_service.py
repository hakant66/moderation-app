# torch_filter/tests/test_service.py
import io
from PIL import Image

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    js = r.json()
    assert js["ok"] is True
    # The fakes are loaded by the fixture
    assert js["image_model"] is True
    assert js["text_model"] is True


def test_predict_text_allows_low_risk(client):
    # Our FakeTextTS returns a very negative logit -> prob ~ 0.0
    r = client.post("/predict/text", json={"text": "Have a great day!"})
    assert r.status_code == 200, r.text
    js = r.json()
    assert "score" in js and 0.0 <= js["score"] <= 1.0
    assert js["label"] == "unsafe_text"
    # With negative logit, score should be tiny
    assert js["score"] < 0.01


def test_predict_image_blocks_high_risk(client):
    # Create a tiny in-memory RGB image (will be resized by service)
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("red.png", buf, "image/png")}
    r = client.post("/predict/image", files=files)
    assert r.status_code == 200, r.text
    js = r.json()
    assert "score" in js and 0.0 <= js["score"] <= 1.0
    assert js["label"] == "unsafe_image"
    # Our FakeImageTS returns a high logit -> prob ~ 1.0
    assert js["score"] > 0.99
