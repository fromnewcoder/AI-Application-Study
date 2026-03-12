#Quick-start: building and using a PromptTemplate
from prompt_template import PromptTemplate, Example, ExampleKind
from prompt_template import ChainOfThoughtConfig, XMLSchema, TemplateRegistry

# ── Build ─────────────────────────────────────────────────────────
tmpl = (
    PromptTemplate(name="my-task", version="1.0")
    .set_role("You are an expert in...")
    .set_system("Analyse the following and produce XML output.")
    .set_task("Input: {user_input}")
    .set_cot(ChainOfThoughtConfig(steps=[
        "Identify key entities.",
        "Assess each entity.",
        "Summarise findings.",
    ]))
    .set_xml_schema(XMLSchema(
        root_tag="result",
        fields=[("label","classification"), ("score","0-100 confidence")],
        required=["label"],
    ))
    .add_example(Example("good input", "<result>...</result>"))
    .add_example(Example("bad input", "<result>WRONG</result>",
        kind=ExampleKind.NEGATIVE,
        explanation="This is wrong because...",
    ))
)

# ── Validate ──────────────────────────────────────────────────────
warnings = tmpl.validate(user_input="test")
assert not warnings, warnings

# ── Render ────────────────────────────────────────────────────────
msgs = tmpl.render(user_input="My actual input")
# msgs = {'system': '...', 'user': '...'}

# ── Use with any client ────────────────────────────────────────────
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=msgs["system"],
    messages=[{"role": "user", "content": msgs["user"]}],
)
