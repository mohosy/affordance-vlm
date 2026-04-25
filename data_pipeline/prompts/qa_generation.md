# Q&A Generation Prompt (v1)

This is the prompt used to generate part-level affordance Q&A pairs from a HOVA-500K annotation. The annotation provides ground-truth grounding so the model is *constrained* to talk about a real region of a real object — it cannot hallucinate the object identity or the affordance location.

**System instructions sent to Gemini 2.5 Pro:**

```
You write training data for a vision-language model that helps robots reason
about how to interact with objects. You are given (a) an image, and (b) a
ground-truth annotation that names the object, the action that can be performed,
and where on the object that action happens.

Your job: generate {N} concise, part-level affordance Q&A pairs that a robot
controller would need to answer correctly.

Hard rules:
- The QUESTION must be answerable purely from the image; do not reference the
  ground-truth annotation in the question. Treat the image as the only context.
- The ANSWER must be consistent with the ground-truth annotation. If the
  annotation says action="grasp" and the affordance region is "middle-right",
  your answer must reflect that.
- Be specific about parts. "the handle", "the trigger", "the rim of the lid" —
  not "the object" or "this thing".
- Use one of these question types:
    1. Identification: "What part of this {object} is highlighted / used to
       perform {action}?"
    2. Localization: "Where on this {object} would you {action} it?"
    3. Substitution: "If the {object} were unavailable, what other tool in
       the room could substitute for {action}? If none is visible, say so."
    4. Mechanism: "Why is the {part} of this {object} shaped this way for
       the action of {action}?"
- Avoid yes/no questions.
- Each answer must be 1-3 short sentences. No padding.

Output format: a single JSON array, no prose around it. Each element:
{
  "question": "...",
  "answer": "...",
  "type": "identification|localization|substitution|mechanism"
}
```

**Per-call user payload:**

```
GROUND TRUTH:
- object: {object_name}
- part (if known): {part_name or "n/a"}
- canonical action: {action or canonical_actions_joined or "interact"}
- affordance region in image: {location_description}
- bbox (if 3doi): {bbox or "n/a"}

[IMAGE attached]

Generate {N} Q&A pairs.
```

The location description comes from `annotations.describe_location()` and is a coarse spatial hint ("middle-left ~26% from left, 46% from top"), not exact pixel coordinates — the goal is to ground the answer, not to teach pixel regression.
