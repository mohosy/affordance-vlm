import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

type Body = {
  model: "gemini" | "claude" | "gpt4o";
  image: string; // data URL
  question: string;
};

const PROMPT_INSTRUCTION =
  "Look at the image and answer the question concisely (1-3 short sentences). " +
  "Be specific about which part of any object you reference.\n\nQuestion: ";

function decodeDataUrl(dataUrl: string): { mediaType: string; base64: string } | null {
  const match = dataUrl.match(/^data:(image\/[a-zA-Z+.-]+);base64,(.+)$/);
  if (!match) return null;
  return { mediaType: match[1], base64: match[2] };
}

function missingKey(key: string) {
  return NextResponse.json(
    { code: "missing_key", key, error: `${key} not set` },
    { status: 400 }
  );
}

export async function POST(req: Request) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const { model, image, question } = body;
  if (!model || !image || !question) {
    return NextResponse.json({ error: "missing model/image/question" }, { status: 400 });
  }
  const decoded = decodeDataUrl(image);
  if (!decoded) {
    return NextResponse.json({ error: "image must be a data URL" }, { status: 400 });
  }
  const userText = PROMPT_INSTRUCTION + question;

  try {
    if (model === "gemini") {
      const apiKey = process.env.GOOGLE_API_KEY;
      if (!apiKey) return missingKey("GOOGLE_API_KEY");
      const { GoogleGenAI } = await import("@google/genai");
      const ai = new GoogleGenAI({ apiKey });
      const result = await ai.models.generateContent({
        model: "gemini-2.5-pro",
        contents: [
          {
            role: "user",
            parts: [
              {
                inlineData: {
                  mimeType: decoded.mediaType,
                  data: decoded.base64,
                },
              },
              { text: userText },
            ],
          },
        ],
        config: { temperature: 0.2 },
      });
      const answer = result.text ?? "";
      return NextResponse.json({ answer });
    }

    if (model === "claude") {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) return missingKey("ANTHROPIC_API_KEY");
      const Anthropic = (await import("@anthropic-ai/sdk")).default;
      const client = new Anthropic({ apiKey });
      const resp = await client.messages.create({
        model: "claude-opus-4-7",
        max_tokens: 400,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: decoded.mediaType as
                    | "image/jpeg"
                    | "image/png"
                    | "image/webp"
                    | "image/gif",
                  data: decoded.base64,
                },
              },
              { type: "text", text: userText },
            ],
          },
        ],
      });
      const block = resp.content[0];
      const answer = block && "text" in block ? block.text : "";
      return NextResponse.json({ answer });
    }

    if (model === "gpt4o") {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) return missingKey("OPENAI_API_KEY");
      const OpenAI = (await import("openai")).default;
      const client = new OpenAI({ apiKey });
      const resp = await client.chat.completions.create({
        model: "gpt-4o",
        max_tokens: 400,
        messages: [
          {
            role: "user",
            content: [
              { type: "image_url", image_url: { url: image } },
              { type: "text", text: userText },
            ],
          },
        ],
      });
      const answer = resp.choices[0]?.message?.content ?? "";
      return NextResponse.json({ answer });
    }

    return NextResponse.json({ error: `unknown model ${model}` }, { status: 400 });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    console.error("predict error", model, message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
