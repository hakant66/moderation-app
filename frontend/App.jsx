import React, { useState } from "react";
import { createRoot } from "react-dom/client";
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
function CategoryChips({ scores }) {
  const entries = Object.entries(scores || {}).sort(([a], [b]) => a.localeCompare(b));
  return (<div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{entries.map(([k, v]) => (<span key={k} style={{ border: "1px solid #ccc", borderRadius: 12, padding: "4px 8px" }}>{k} : {Number.isFinite(v) ? v.toFixed(2) : String(v)}</span>))}</div>);
}
function App() {
  const [text, setText] = useState("");
  const [imgFile, setImgFile] = useState(null);
  const [imgUrl, setImgUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const moderateText = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/moderate/text`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) { setError(e.message || String(e)); } finally { setLoading(false); }
  };
  const moderateImage = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const fd = new FormData();
      if (imgFile) fd.append("file", imgFile);
      if (imgUrl) fd.append("image_url", imgUrl);
      const res = await fetch(`${API_BASE}/api/moderate/image`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) { setError(e.message || String(e)); } finally { setLoading(false); }
  };
  return (<div style={{ maxWidth: 880, margin: "32px auto", fontFamily: "system-ui, sans-serif" }}>
    <h1>Moderation Demo</h1>
    <p>Checks text and images using OpenAI Moderation with an optional PyTorch prefilter.</p>
    <section style={{ margin: "24px 0", padding: 16, border: "1px solid #eee", borderRadius: 12 }}>
      <h2>Text</h2>
      <textarea style={{ width: "100%", minHeight: 120 }} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste text here..." />
      <div style={{ marginTop: 8 }}><button onClick={moderateText} disabled={loading || !text.trim()}>Check Text</button></div>
    </section>
    <section style={{ margin: "24px 0", padding: 16, border: "1px solid #eee", borderRadius: 12 }}>
      <h2>Image</h2>
      <input type="file" accept="image/*" onChange={(e) => setImgFile(e.target.files?.[0] || null)} />
      <div style={{ marginTop: 8 }}>or image URL:</div>
      <input style={{ width: "100%" }} placeholder="https://..." value={imgUrl} onChange={(e) => setImgUrl(e.target.value)} />
      <div style={{ marginTop: 8 }}><button onClick={moderateImage} disabled={loading || (!imgFile && !imgUrl)}>Check Image</button></div>
    </section>
    {loading && <div>Checking…</div>}
    {error && <div style={{ color: "crimson" }}>Error: {error}</div>}
    {result && (<section style={{ padding: 16, border: "1px solid #eee", borderRadius: 12 }}>
      <h2>Result: {result.decision?.toUpperCase?.() || "UNKNOWN"}</h2>
      {result.reasons?.length ? <p>Reasons: {result.reasons.join(", ")}</p> : <p>No rule was triggered.</p>}
      <CategoryChips scores={result.scores} />
      {result.actions?.length ? (<div style={{ marginTop: 12 }}><strong>Suggested actions: </strong>{result.actions.join(", ")}</div>) : null}
      {result.hash ? (<div style={{ marginTop: 12, fontSize: 12, color: "#666" }}>hash: {String(result.hash).slice(0, 12)}…</div>) : null}
    </section>)}
  </div>);
}
createRoot(document.getElementById("root")).render(<App />);
