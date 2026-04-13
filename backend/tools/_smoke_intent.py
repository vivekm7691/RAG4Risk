import sys

sys.path.insert(0, ".")
from app.services.query_intent_service import strip_reasoning_traces, extract_json_object, QueryIntentService

wrap = "<" + "think" + ">" + "reasoning here" + "<" + "/" + "think" + ">"
payload = '{"intent_summary":"x","document_weights":{"statement of work":0.2,"solution description document":0.2,"proposal document":0.2,"risk register":0.2,"issue log":0.2}}'
t = wrap + payload
assert strip_reasoning_traces(t).strip() == payload
assert extract_json_object(t) == payload
svc = QueryIntentService()
r = svc.process_llm_output(t, None)
assert not r.used_fallback
assert abs(sum(r.document_weights.values()) - 1.0) < 1e-6
print("ok")
