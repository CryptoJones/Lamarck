# Tier 3 held-out test set

100 JSON-mode problems used to score the fine-tuned tiny student
model that Tier 3 produces from G_N's curriculum. **G_N MUST
NEVER SEE this directory** - that is the load-bearing property
that keeps Tier 3 gaming-resistant.

## Files

- `json_mode_problems.jsonl` - the held-out corpus (100 lines).
  Each line: `{"input": "<prompt>", "schema": <JSON-Schema>}`.

## Regenerating

The corpus is deterministic. To re-emit it byte-identically:

```bash
python scripts/eval/generate_tier3_holdout.py
```

The seed (`0x42ACAC1A` in the generator) is locked under v1 -
changing it is a v2 event. The generator covers 10 domain
templates (10 problems each):

| Template | Domain |
|---|---|
| `gen_user_profile`         | User profile w/ name+email+age+country |
| `gen_sensor_reading`       | Sensor metrics w/ device, unit enum    |
| `gen_dictionary_definition`| Lexicon entry w/ POS enum + examples   |
| `gen_product_listing`      | Catalog row w/ SKU pattern, category   |
| `gen_calendar_event`       | Event w/ attendees array of emails     |
| `gen_geocoded_location`    | City w/ lat/lon bounds, population int |
| `gen_config_block`         | Service config w/ port range + env enum|
| `gen_recipe`               | Recipe w/ ingredients array of objects |
| `gen_contact`              | Contact card w/ phones + nested address|
| `gen_system_status`        | Status w/ component + status enums     |

## Why these domains

Diversity stress-tests the student's ability to map natural-language
requests onto structured outputs across familiar shapes (user
profiles, products), nested shapes (recipes' ingredients, contacts'
addresses), and constraint-heavy shapes (sensor unit enums, config
port ranges). A curriculum that overfits to one of these shapes
will under-score relative to one that teaches general JSON-mode
discipline.

## Schema discipline

Every schema must declare `type=object` and at least one entry in
its `required` list. Empty-object-passes-everything would dilute
the eval signal; the corpus-validation test enforces this.
